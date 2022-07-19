# bin/bash

python -m torch.distributed.launch --nproc_per_node 1 --master_port 8889 main.py \
--dataset 'millionAID' --model 'vitae_win' --exp_num 1 --batch-size 8 --epochs 300 \
--img_size 224 --split 100 --lr 5e-4  --weight_decay 0.05 --gpu_num 1 --lbp_root 'cyclic LBP 10' --output test_run3