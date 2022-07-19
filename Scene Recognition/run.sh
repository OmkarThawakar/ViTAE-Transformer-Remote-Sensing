#!/bin/bash                                                                    
#SBATCH --job-name=LBP5
#SBATCH -w p3-r52-b.g42cloud.net                                                 
#SBATCH --ntasks=1                                                             
#SBATCH --output=logs/logs_LBP_10_%j.txt                                       
#SBATCH --partition=multigpu  
#SBATCH --time=3-00:00:00                                                      
#SBATCH --cpus-per-task=80                                                     
#SBATCH --gpus=16   
#SBATCH --nodes=1                                        
# Run the hellompi program with mpirun. The -n flag is not required;           
# mpirun will automatically figure out the best configuration from the         
# Slurm environment variables.  


python -m torch.distributed.launch --nproc_per_node 16 --master_port 8889 main.py \
--dataset 'millionAID' --model 'vitae_win' --exp_num 1 --batch-size 16 --epochs 300 \
--img_size 224 --split 100 --lr 5e-4  --weight_decay 0.05 --gpu_num 16 --lbp_root 'cyclic LBP 10'  --output test_run_cyclic_LBP_10 >> ./logs/logs_LBP_10.out
    
echo "Completed !!!"