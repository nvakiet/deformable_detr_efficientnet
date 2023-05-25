from util.plot_utils import plot_logs, plot_precision_recall, plot_mAP
from pathlib import Path
import matplotlib.pyplot as plt
import os

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


def get_file_path(dir_names):
    result = []
    for dir in dir_names:
        path = os.path.join(".", "exps", dir)
        result.append(Path(path))
    return result


model_path = "./exps/effi_v2s_deformable_detr/"
model_path = "./exps/mobilenet_v3_deformable_detr/"
model_path = "./exps/resnet_deformable_detr/"
# with open(model_path + "log.txt", 'r') as fp:
#     num_line = len(fp.readlines())

compare_path = "./exps/resnet_deformable_detr3"
compare_path = "./exps/original"


list_path = ["original",
             "mobilenet_v3_deformable_detr",
             "mobilenet_v3_deformable_detr_b2",
             "resnet_deformable_detr_lr1e-4",
             "resnet_deformable_detr_lr5e-5"
             ]

list_path = ["original",
             "effi_v2s_deformable_detr",
             "mobilenet_v3_deformable_detr_b2",
             "resnet_deformable_detr_lr1e-4_b2"]

list_path = ["original",
             "resnet_deformable_detr_lr2e-4",
             "resnet_deformable_detr_lr1e-4",
             "resnet_deformable_detr_lr5e-5",
             ]


list_path = ["original",
             "resnet_deformable_detr_lr1e-4_b2"
             ]

list_path = ["original",
             "resnet_deformable_detr_lr1e-4_b2",
             "mobilenet_v3_deformable_detr_b2",
             "effi_v2s_deformable_detr",
             "swin_deformable_detr"
             ]
file = get_file_path(list_path)
num_epoch = 50


fields = ("loss", "class_error", "loss_bbox", "loss_giou")
fig, _ = plot_logs(logs=file, fields=fields, num_epoch=num_epoch)
fig.savefig("result.png")


fig, _ = plot_mAP(logs=file, num_epoch=num_epoch)
fig.savefig("result_map.png")
