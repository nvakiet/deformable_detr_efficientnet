#!/usr/bin/env bash

set -x

python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/effi_v2s_deformable_detr2" \
    --backbone "efficientnet" \
    --batch_size 2