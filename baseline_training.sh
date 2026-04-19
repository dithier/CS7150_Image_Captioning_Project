#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=8:00:00
#SBATCH --job-name=baseline_training
#SBATCH --output=logs/baseline_v2_adam/%j.out
#SBATCH --error=logs/baseline_v2_adam/%j.err

module load cuda/12.1.1 anaconda3/2024.06
source activate cs7150

export PYTHONPATH=$(pwd)

python -u baseline/adam_baseline_training_v2.py \
	--epochs 30 \
	--checkpoint False \
	--dataset_dir flickr8k \
	--log_dir runs/baseline_v2_adam \
    --save_path saved_models/baseline_v2_adam/
