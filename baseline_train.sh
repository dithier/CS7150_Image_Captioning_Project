#!/bin/bash
#SBATCH -p courses-gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=baseline_training
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load cuda/12.1.1 anaconda3/2024.06
source activate cs7150

python baseline_training.py \
	--epochs 30 \
	--checkpoint False \
	--dataset_dir flickr8k
	--log_dir runs/first_pass_3_14 \
        --save_path saved_models/models_3_14
