#!/usr/bin/env bash

set -x

python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/haha" \
    --batch_size 2 