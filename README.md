# Set up environment

## create conda environment

```bash
conda create -n env_name python=3.8
```

## activate conda environment
```bash
conda activate env_name
```

## install dependency
```bash
pip install -r requirements.txt
```

## install MSDeformAttn package (required)

```bash
pip install models/ops
```

# Download coco dataset
- http://images.cocodataset.org/zips/train2017.zip
- http://images.cocodataset.org/zips/val2017.zip
- http://images.cocodataset.org/annotations/annotations_trainval2017.zip

After download unzip all in **path/to/coco**\
Folder structure should look like this

```
path/to/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

# Training
```bash
bash train_efficientnet.sh
```

content of **train_efficientnet.sh**
```
#!/usr/bin/env bash

set -x

python -u main.py \
    --num_feature_levels 1 \
    --output_dir "exps/effi_v2s_deformable_detr" \
    --backbone "efficientnet" \
    --batch_size 2 \
    --lr 1e-4 \
    --lr_backbone 1e-5
```
> **Note**
> Use arg **--coco_path "path/to/coco"** to update coco path\
**path/to/coco** is coco dataset directory in previous step