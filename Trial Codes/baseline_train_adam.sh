#!/bin/bash
#SBATCH -p courses-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=baseline_training_adam
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

#module load cuda/12.1.1 anaconda3/2024.06
#source activate cs7150

mkdir -p logs runs/adam_pass_1 saved_models/adam_pass_1

python baseline_training_adam.py \
	--epochs 30 \
	--checkpoint False \
	--dataset_dir flickr8k \
	--log_dir runs/adam_pass_1 \
	--save_path saved_models/adam_pass_1/ 