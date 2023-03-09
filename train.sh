python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/effi_v2s_deformable_detr" \
    --backbone "efficientnet" \
    --resume "deformable_detr_weights/r50_deformable_detr_single_scale-checkpoint.pth" \
    --batch_size 1
