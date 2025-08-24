import os
# import faiss
import torch
import torch.nn as nn
import shutil
import numpy as np

from tqdm import tqdm
from models import SFDA
from datasets import SFDADataset
from torch.utils.data.dataloader import DataLoader

from argparase import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy, inference

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import norm



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']  # copy initial learning rate
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def identify_source_private(args, model, dataloader):
    model.eval()

    all_fea, all_output, _ = inference(args, model, dataloader, apply_softmax=True)

    max_probs, predict = torch.max(all_output, 1)
    confident_mask = max_probs > args.w_0
    confident_predict = predict[confident_mask]
    class_counts = torch.bincount(confident_predict, minlength=args.class_num)

    # class_counts = torch.bincount(predict)  # Count occurrences of each predicted class
    for class_idx, count in enumerate(class_counts):
        args.logger.info(f"class {class_idx}: {count.item()} samples")
    
    threshold = 0.01 * len(dataloader.dataset) / args.class_num 
    args.logger.info(f"open-close: threshold: {threshold}")
    
    # Determine if source-private classes exist
    source_private_class_exists = torch.min(class_counts)< threshold
    
    if not source_private_class_exists:
        args.s_class_num = args.class_num
        args.mode = 'full'
        args.logger.info("Adaptation under Full mode")
    else:
        args.mode = 'partial'
        args.logger.info(f"Adaptation under Partial mode")


def obtain_pseudo_label(args, epochs, model, dataloader):
    model.eval()

    all_fea, all_output, all_label = inference(args, model, dataloader, apply_softmax=True) 
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float().cpu() == all_label).item() / float(all_label.size()[0])

    all_fea_n = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    # identify target-private
    ent = torch.sum(-all_output * torch.log(all_output + 1e-6), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu().numpy()

    gmm_en = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm_en.fit(ent.reshape(-1, 1))

    sorted_probs, sorted_indices = torch.sort(all_output, dim=1, descending=True)
    first_confidence = sorted_probs[:, 0].cpu().numpy()
    second_confidence = sorted_probs[:, 1].cpu().numpy()
    confidence_diff = first_confidence - second_confidence

    labels = gmm_en.predict(ent.reshape(-1, 1))
    ENT_THRESHOLD = gmm_en.means_.mean()

    # determine where target_private exist
    if epochs == 0 and args.mode=='partial':
        means = gmm_en.means_.flatten()
        weights = gmm_en.weights_
        sorted_idx = np.argsort(means)
        args.logger.info(f"weights:{weights[sorted_idx[0]]:.3f}, means[0]:{means[sorted_idx[0]]:.3f}, means[1]:{means[sorted_idx[1]]:.3f}")
        ent_threshold = weights[sorted_idx[0]] / (means[sorted_idx[1]]**2 - means[sorted_idx[0]]**2)
        args.logger.info(f"ent_threshold:{ent_threshold:.3f}")
        if ent_threshold > args.beta:  
            args.mode = 'partial-set'
            args.logger.info(" adaption under partial-set")

    idx = np.where(labels == 1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1

    class_counts = torch.bincount(predict)  # Count occurrences of each predicted class

    # calibrate common_class number & known samples
    if args.mode == 'partial-set':    
        known_idx = np.arange(len(all_label))
        source_pre = (class_counts < (0.5 * (len(known_idx) / args.class_num))).sum().item()  
        shared_class_pred = args.class_num - source_pre  
    elif args.mode == 'partial': 
        source_pre = (class_counts < (0.5 * (len(known_idx) / args.class_num))).sum().item()  
        shared_class_pred = args.class_num - source_pre  
        
    else: 
        known_idx = np.where(labels != iidx)[0]
        known_idx_1 = np.where(labels != iidx)[0]
        known_idx_2 = np.where(confidence_diff > 0.6)[0]
        known_idx = np.intersect1d(known_idx_1, known_idx_2)
        shared_class_pred = args.class_num
        args.s_class_num = shared_class_pred

    # Update the number of shared classes
    if shared_class_pred > args.s_class_num:
        args.s_class_num = shared_class_pred
        args.logger.info(f"Predicted shared class number: {args.s_class_num:.1f}")

    all_fea_n = all_fea_n[known_idx, :]
    all_fea = all_fea[known_idx, :] 
    all_output = all_output[known_idx, :]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]

    all_fea_n = all_fea_n.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea_n)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]

    dd = cdist(all_fea_n, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    Magnitude = torch.norm(all_fea, p=2, dim=1) ** 2 
    Knowns_Magnitude = torch.max(Magnitude)
    # print(Knowns_Magnitude)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea_n)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea_n, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label

    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    # log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy * 100, acc * 100)
    log_str = 'Pseudo Accuracy = {:.2f}%, Threshold = {:.2f},'.format(acc * 100, ENT_THRESHOLD)
    args.logger.info(log_str)

    return guess_label.astype('int'), Knowns_Magnitude


def train(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)

    mem_label, Minimum_Knowns_Magnitude = obtain_pseudo_label(args, epoch_idx, model, test_dataloader)

    model.train()

    all_pred_loss_stack = []
    for imgs_train, _, imgs_label, imgs_idx in tqdm(train_dataloader, ncols=60):

        iter_idx += 1
        imgs_train = imgs_train.to(args.device)

        pred = mem_label[imgs_idx]
        features_test, outputs_test = model(imgs_train, apply_softmax=False)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        pred = torch.tensor(pred).to(args.device)
        imgs_idx = imgs_idx.to(args.device)
        
        known_mask = pred < args.s_class_num
        outputs_test_known = outputs_test[known_mask, :]
        features_test_known = features_test[known_mask, :]
        
        known_inx = imgs_idx[known_mask]
        pred = mem_label[known_inx.cpu().numpy()]

        if args.mode == 'partial-set':    
            outputs_test_unknown = torch.empty((0, outputs_test.shape[1]), device=args.device)
            features_test_unknown = torch.empty((0, features_test.shape[1]), device=args.device)
        else:
            outputs_test_unknown = outputs_test[~known_mask, :]
            features_test_unknown = features_test[~known_mask, :]

        C_entropy_loss = nn.CrossEntropyLoss()(outputs_test_known, torch.from_numpy(pred).long().to(args.device))

        softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
        entropy_loss_known = torch.mean(Entropy(args, softmax_out_known))
        softmax_out_unknown = nn.Softmax(dim=1)(outputs_test_unknown)
        entropy_loss_un = torch.mean(Entropy(args, softmax_out_unknown, open_flag=True))
        # print(entropy_loss_un)

        feat_loss_un = torch.norm(features_test_unknown, p=2, dim=1) ** 2
        feat_loss_known = torch.norm(features_test_known, p=2, dim=1) ** 2 
        cicular_loss_un = feat_loss_un.mean()
        loss_known = torch.relu(Minimum_Knowns_Magnitude - feat_loss_known)
        cicular_loss_known = loss_known.mean()
        circular_loss = 0.0001 * (cicular_loss_known + cicular_loss_un)

        uncertainty_loss = entropy_loss_known + args.lam * entropy_loss_un

        loss = args.lam * C_entropy_loss + uncertainty_loss + circular_loss
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_pred_loss_stack.append(loss.cpu().item())

    # print("all_pred_loss_stack:", all_pred_loss_stack)
    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)

    return train_loss_dict


@torch.no_grad()
def test(args, model, dataloader, src_flg=False):
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        imgs_test = imgs_test.to(args.device)
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0)  # [N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0)  # [N, C]

    h_score, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg,
                                                         open_thresh=args.w_0)
    return h_score, known_acc, unknown_acc


def main(args):
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "mps")
    print(f"Using device: {args.device}")
    this_dir = os.path.join(os.path.dirname(__file__), ".")

    model = SFDA(args)  

    model = model.to(args.device)

    save_dir = os.path.join(this_dir, "checkpoints", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, "{}".format(args.source_train_type))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt")

    if args.reset:
        raise ValueError

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])  # parameter weights initialization
    else:
        print(args.checkpoint)
        raise ValueError("YOU MUST SET THE APPROPORATE SOURCE CHECKPOINT FOR TARGET MODEL ADPTATION!!!")

    shutil.copy("./train_target_da.py", os.path.join(args.save_dir, "train_target.py"))
    # shutil.copy("./utils/net_utils.py", os.path.join(args.save_dir, "net_utils.py"))

    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]  # keep the backbone layer almost the same 0.1

    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]  # update feature embedded layer   1

    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False  # fix classifier layer

    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)  # copy learning rate

    target_data_list = open(os.path.join(args.target_data_dir, "image_unida_list.txt"), "r").readlines()
    target_dataset = SFDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)

    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)  

    notation_str = "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="

    args.logger.info(notation_str)

    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0

    # determine source_private
    identify_source_private(args, model, target_test_dataloader)

    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # Train on target
        loss_dict = train(args, model, target_train_dataloader, target_test_dataloader,
                                         optimizer, epoch_idx)

        args.logger.info(
            "Epoch: {}/{}, train_all_loss:{:.3f},\n".format(epoch_idx + 1, args.epochs, loss_dict["all_pred_loss"], ))

        checkpoint_file = "latest_target_checkpoint.pth"
        torch.save({
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict()}, os.path.join(save_dir, checkpoint_file))

        # Evaluate on target
        hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)
        args.logger.info(
            "Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))

        if args.target_label_type == "PDA":
            if knownacc >= best_known_acc:
                best_epoch_idx = epoch_idx
                best_known_acc = knownacc

        else:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx

                checkpoint_file = "{}_best_target_checkpoint.pth".format(args.dataset)
                torch.save({
                    "epoch":epoch_idx,
                    "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))

        args.logger.info(
            "Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc,
                                                                                 best_unknown_acc))


if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)

    args.checkpoint = os.path.join("checkpoints", args.dataset, "source_{}".format(args.s_idx), \
                                   "source_{}_{}".format(args.source_train_type, args.target_label_type),
                                   "latest_source_checkpoint.pth")
    args.reset = False
    args.s_class_num = 0
    main(args)
