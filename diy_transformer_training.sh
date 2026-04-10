#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --job-name=transformer_training
#SBATCH --output=logs/transformer/%j.out
#SBATCH --error=logs/transformer/%j.err

# Note: may need to change params in file run to run

module load cuda/12.1.1 anaconda3/2024.06
source activate cs7150

python -u ViT/diy_transformer_training_2.py \
	--epochs 30 \
    --print_freq 50 \
	--checkpoint False \
	--dataset_dir flickr8k \
	--log_dir runs/transformer \
    --save_path saved_models/transformer
    
