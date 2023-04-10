from models import build_model
import argparse
from main import get_args_parser
from torchinfo import summary
import torch
"""
parser = argparse.ArgumentParser(
    'Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


args.num_feature_levels = 4
model, _, _ = build_model(args)
# summary(model, input_size=(1, 3, 450, 613), depth=100)
print(model)
"""


checkpoint = torch.load(
    "exps/resnet_deformable_detr3/checkpoint0001.pth", map_location='cpu')
print(checkpoint.keys())
for k in checkpoint.keys():

    print("=" * 100)
    print(k)
    if k == "model":
        print("ignore")
        continue
    try:
        print(checkpoint[k].keys())
    except:
        print("no keys")

print("*" * 100)
print("epoch", checkpoint["epoch"], "\n")
# print("args", checkpoint["args"])
for k in checkpoint["lr_scheduler"].keys():
    print(str(k).ljust(30, " "), checkpoint["lr_scheduler"][k])
