#!/bin/bash                                                                    
#SBATCH --job-name=L40
#SBATCH -w p4-r68-b.g42cloud.net                                                 
#SBATCH --ntasks=1                                                             
#SBATCH --output=logs/logs_LBP_%j.txt                                       
#SBATCH --partition=multigpu  
#SBATCH --time=3-00:00:00                                                      
#SBATCH --cpus-per-task=80                                                     
#SBATCH --gpus=16   
#SBATCH --nodes=1                                        
# Run the hellompi program with mpirun. The -n flag is not required;           
# mpirun will automatically figure out the best configuration from the         
# Slurm environment variables.  


python -m torch.distributed.launch --nproc_per_node 16 --master_port 8887 main.py \
--dataset 'millionAID' --model 'vitae_win' --exp_num 1 --batch-size 16 --epochs 300 \
--img_size 224 --split 100 --lr 5e-4  --weight_decay 0.05 --gpu_num 16 --lbp_root 'cyclic LBP' --output test_run_cyclic_LBP >> ./logs/logs_LBP.out
    
echo "Completed !!!"