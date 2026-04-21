#!/bin/bash
#SBATCH -p courses-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=08:00:00
#SBATCH --job-name=vit_transformer_resume
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda/12.1.1 anaconda3/2024.06
source activate cs7150

mkdir -p logs saved_models/vit_transformer_pass_3 runs/vit_transformer_pass_3 

python pretrained_transformer_training.py \
    --epochs 30 \
    --checkpoint False \
    --dataset_dir flickr8k \
    --log_dir runs/vit_transformer_pass_3 \
    --save_path saved_models/vit_transformer_pass_3/ \
    --weight_decay 1e-4 \
    --print_freq 250
