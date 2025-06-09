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


def train(args, model, train_dataloader, test_dataloader, optimizer, feature_bank, score_bank, epoch_idx=0.0):
    model.train()
    all_pred_loss_stack = []

    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    for imgs_train, _, imgs_label, imgs_idx in tqdm(train_dataloader, ncols=60):

        iter_idx += 1
        imgs_train = imgs_train.to(args.device)
        features_test, outputs_test = model(imgs_train, apply_softmax=False)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        alpha = (1 + 10 * iter_idx / iter_max) ** (-1)  # 1 -> 0

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.detach().clone()

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            pred_bs = softmax_out

            feature_bank[imgs_idx] = output_f_.detach().clone()  # update known samples
            score_bank[imgs_idx] = pred_bs.detach().clone()

            distance = output_f_ @ feature_bank.T  # matrix dot product distance in feature space with known samples and the batch
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=3 + 1)  # compare topk in whole samples
            idx_near = idx_near[:, 1:]  # batch x K    # filter out self
            score_near = score_bank[idx_near]  # preditive distributions of top-k samples batch x K x C

            fea_near = feature_bank[idx_near]  # batch x K x num_dim

        # nn
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 3, -1)

        loss = torch.mean((F.kl_div(softmax_out_un,  # initial out
                                    score_near,  # top K nearest samples  batch x K x C
                                    reduction='none').sum(-1)).sum(1))

        mask = torch.ones((imgs_train.shape[0], imgs_train.shape[0]))  # [batch batch]
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag  # set diag element to 0
        copy = softmax_out.T  # The whole batch

        dot_neg = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg * mask.to(args.device)).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha

        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_pred_loss_stack.append(loss.cpu().item())

    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)

    return train_loss_dict, feature_bank, score_bank


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
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "mps")
    print(f"Using device: {args.device}")
    this_dir = os.path.join(os.path.dirname(__file__), ".")

    model = SFDA(args)  # the same model architecture

    model = model.to(args.device)

    save_dir = os.path.join(this_dir, "checkpoints_aad", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type,"{}".format(args.source_train_type))

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

    # shutil.copy("./train_target.py", os.path.join(args.save_dir, "train_target.py"))
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
                                        num_workers=args.num_workers, drop_last=False)  # the same dataset

    notation_str = "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="

    args.logger.info(notation_str)

    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0

    num_sample = len(target_test_dataloader.dataset)

    # building feature bank and score bank
    score_bank = torch.zeros(num_sample, args.class_num).to(args.device)
    feature_bank = torch.zeros(num_sample, args.embed_feat_dim).to(args.device)

    # AaD
    model.eval()
    all_fea, all_output, _ = inference(args, model, target_test_dataloader, apply_softmax=True)

    all_fea = F.normalize(all_fea)
    feature_bank = all_fea.detach().clone().to(args.device)
    score_bank = all_output.detach().clone().to(args.device)  # .cpu()

    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # Train on target
        loss_dict, feature_bank, score_bank = train(args, model, target_train_dataloader, target_test_dataloader,
                                                    optimizer, feature_bank, score_bank, epoch_idx)
        args.logger.info(
            "Epoch: {}/{}, train_all_loss:{:.3f},\n".format(epoch_idx + 1, args.epochs, loss_dict["all_pred_loss"], ))

        # checkpoint_file = "latest_target_checkpoint.pth"
        # torch.save({
        #     "epoch": epoch_idx,
        #     "model_state_dict": model.state_dict()}, os.path.join(save_dir, checkpoint_file))

        # Evaluate on target
        hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)
        args.logger.info(
            "Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))

        if hscore >= best_h_score:
            best_h_score = hscore
            best_known_acc = knownacc
            best_unknown_acc = unknownacc
            best_epoch_idx = epoch_idx

            # checkpoint_file = "{}_best_target_checkpoint.pth".format(args.dataset)         
            # torch.save({
            #         "epoch":epoch_idx,
            #         "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))


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
    main(args)