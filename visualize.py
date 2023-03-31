from util.plot_utils import plot_logs, plot_precision_recall
from pathlib import Path
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
################# UNIQUE ################
train_lr
train_grad_norm
test_coco_eval_bbox
epoch
n_parameters

################# OTHER #################
class_error
loss
loss_ce
loss_bbox
loss_giou
loss_ce_0
loss_bbox_0
loss_giou_0
loss_ce_1
loss_bbox_1
loss_giou_1
loss_ce_2
loss_bbox_2
loss_giou_2
loss_ce_3
loss_bbox_3
loss_giou_3
loss_ce_4
loss_bbox_4
loss_giou_4
loss_ce_unscaled
class_error_unscaled
loss_bbox_unscaled
loss_giou_unscaled
cardinality_error_unscaled
loss_ce_0_unscaled
loss_bbox_0_unscaled
loss_giou_0_unscaled
cardinality_error_0_unscaled
loss_ce_1_unscaled
loss_bbox_1_unscaled
loss_giou_1_unscaled
cardinality_error_1_unscaled
loss_ce_2_unscaled
loss_bbox_2_unscaled
loss_giou_2_unscaled
cardinality_error_2_unscaled
loss_ce_3_unscaled
loss_bbox_3_unscaled
loss_giou_3_unscaled
cardinality_error_3_unscaled
loss_ce_4_unscaled
loss_bbox_4_unscaled
loss_giou_4_unscaled
cardinality_error_4_unscaled


'''
model_path = "./exps/resnet_deformable_detr/"
model_path = "./exps/effi_v2s_deformable_detr/"
with open(model_path + "log.txt", 'r') as fp:
    num_line = len(fp.readlines())

file = [model_path, "./exps/original"]
file = [Path(f) for f in file]
fields = ("loss", "class_error", "loss_bbox", "loss_giou")
fig, _ = plot_logs(logs=file, fields=fields, num_epoch=num_line)
fig.savefig("result.png")

fields = ("mAP",)
fig, _ = plot_logs(logs=file, fields=fields, num_epoch=num_line)
fig.savefig("result_map.png")
