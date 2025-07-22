
echo "PDA ON Office"
python train_target_DADA.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --target_label_type PDA --lam 1.0 --alpha 1 --beta 1.10   
python train_target_DADA.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001 --target_label_type PDA --lam 1.0 --alpha 1 --beta 1.10
python train_target_DADA.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001 --target_label_type PDA --lam 1.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001 --target_label_type PDA --lam 1.0 --alpha 1 --beta 1.10
python train_target_DADA.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001 --target_label_type PDA --lam 1.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001 --target_label_type PDA --lam 1.0 --alpha 1 --beta 1.10

echo "PDA ON Office-Home"
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001 --target_label_type PDA --lam 2.0 --alpha 1 --beta 0.80

echo "PDA ON VisDA"
python train_target_DADA.py --backbone_arch resnet101 --lr 0.0001 --dataset VisDA --target_label_type PDA --s_idx 0 --t_idx 1 --lam 1.0 --alpha 1 --beta 0.80 --epochs 30

