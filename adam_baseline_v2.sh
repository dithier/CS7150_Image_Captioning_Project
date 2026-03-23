#!/bin/bash
#SBATCH -p courses-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=baseline_training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda/12.1.1 anaconda3/2024.06
source activate cs7150

python adam_baseline_training_v2.py \
	--epochs 30 \
	--checkpoint False \
	--dataset_dir flickr8k \
	--log_dir runs/baseline_v2_adam \
    --save_path saved_models/baseline_v2_adam/
