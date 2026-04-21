#!/bin/bash

mkdir -p logs runs/diy_transformer_pass_1 saved_models/diy_transformer_pass_1

python diy_transformer_training.py \
    --epochs 30 \
    --checkpoint False \
    --dataset_dir flickr8k \
    --lr 1e-4 \
    --log_dir runs/diy_transformer_pass_1 \
    --save_path saved_models/diy_transformer_pass_1/