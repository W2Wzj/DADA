
echo "OPDA ON Office"
python train_target_DADA.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --target_label_type OPDA --lam 0.3 --alpha 2 --beta 1.10   
python train_target_DADA.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001 --target_label_type OPDA --lam 0.3 --alpha 2 --beta 1.10
python train_target_DADA.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001 --target_label_type OPDA --lam 0.3 --alpha 2 --beta 1.10
python train_target_DADA.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001 --target_label_type OPDA --lam 0.3 --alpha 2 --beta 1.10
python train_target_DADA.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001 --target_label_type OPDA --lam 0.3 --alpha 2 --beta 1.10
python train_target_DADA.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001 --target_label_type OPDA --lam 0.3 --alpha 2 --beta 1.10

echo "OPDA ON Office-Home"
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001 --target_label_type OPDA --lam 2.0 --alpha 2 --beta 0.80

echo "OPDA ON VisDA"
python train_target_DADA.py --backbone_arch resnet101 --lr 0.0001 --dataset VisDA --target_label_type OPDA --lam 1.0 --alpha 1 --beta 0.90 --epochs 30
