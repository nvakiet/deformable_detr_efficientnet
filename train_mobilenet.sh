#!/usr/bin/env bash

set -x


python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/mobilenet_v3_deformable_detr_b2" \
    --backbone "mobilenet" \
    --batch_size 2 \
    --resume "exps/mobilenet_v3_deformable_detr_b2/checkpoint0007.pth"