from models import build_model
import argparse
from main import get_args_parser
from torchinfo import summary

parser = argparse.ArgumentParser(
    'Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


args.num_feature_levels = 4
model, _, _ = build_model(args)
# summary(model, input_size=(1, 3, 450, 613), depth=100)
print(model)
