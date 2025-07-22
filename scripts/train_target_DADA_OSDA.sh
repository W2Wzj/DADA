
echo "OSDA on Office"
python train_target_DADA.py --dataset Office --s_idx 0 --t_idx 1 --lr 0.001 --target_label_type OSDA --lam 0.3  
python train_target_DADA.py --dataset Office --s_idx 0 --t_idx 2 --lr 0.001 --target_label_type OSDA --lam 0.3  
python train_target_DADA.py --dataset Office --s_idx 1 --t_idx 0 --lr 0.001 --target_label_type OSDA --lam 0.3  
python train_target_DADA.py --dataset Office --s_idx 1 --t_idx 2 --lr 0.001 --target_label_type OSDA --lam 0.3  
python train_target_DADA.py --dataset Office --s_idx 2 --t_idx 0 --lr 0.001 --target_label_type OSDA --lam 0.3  
python train_target_DADA.py --dataset Office --s_idx 2 --t_idx 1 --lr 0.001 --target_label_type OSDA --lam 0.3  

echo "OSDA ON Office-Home"
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 1 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 2 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 0 --t_idx 3 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 0 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 2 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 1 --t_idx 3 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 0 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 1 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 2 --t_idx 3 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 0 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 1 --lr 0.001 --target_label_type OSDA --lam 2.0 
python train_target_DADA.py --dataset OfficeHome --s_idx 3 --t_idx 2 --lr 0.001 --target_label_type OSDA --lam 2.0 

echo "OSDA ON VisDA"
python train_target_DADA.py --backbone_arch resnet101 --s_idx 0 --t_idx 1 --lr 0.0001 --dataset VisDA --target_label_type OSDA --lam 1.0  --epochs 30
