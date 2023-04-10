#!/usr/bin/env bash

set -x

python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/resnet_deformable_detr3" \
    --batch_size 2\
    --lr 1e-4 \
    --lr_backbone 1e-5
