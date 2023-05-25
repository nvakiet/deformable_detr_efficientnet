#!/usr/bin/env bash

set -x

python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/swin_deformable_detr" \
    --backbone "swin" \
    --batch_size 2 \
    --lr_backbone 2e-5 \
    --device 'cuda:1'