#!/bin/bash
#SBATCH --job-name=EVENTOCR                       
#SBATCH -N 1                           
#SBATCH --gres=gpu:3               
#SBATCH -o log/%j.log 
#SBATCH -e log/%j.err  
#SBATCH -p GPUSCS01 
#SBATCH --constraint="python"     

source activate bliva

NAME="ESTR-CoT"
PORT=1025
export CUDA_VISIBLE_DEVICES=1
time=$(date +%s)
torchrun --nproc_per_node=1 --master_port=$PORT train.py \
    --cfg-path train_configs/finetune_bliva_vicuna.yaml \
    > logs/${NAME}_$time.log 2>&1 &

echo "logs/${NAME}_$time.log"
