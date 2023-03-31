import torch
import pandas as pd
from pathlib import Path
from util.plot_utils import plot_precision_recall


def load_eval(eval_path):
    data = torch.load(eval_path)
    # precision is n_iou, n_points, n_cat, n_area, max_det
    precision = data['precision']
    # take precision for all classes, all areas and 100 detections
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    CLASSES = [c for c in CLASSES if c != 'N/A']
    area = 0
    return pd.DataFrame.from_dict({c: p for c, p in zip(CLASSES, precision[0, :, :, area, -1].mean(0) * 100)}, orient='index')


path = "./exps/effi_v2s_deformable_detr/"
data = load_eval(path + '/eval/latest.pth')
print(data.to_string())
