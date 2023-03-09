# from models.backbone import Backbone
import torchvision
from torchinfo import summary
import torch
from torchvision.models.efficientnet import _efficientnet_conf, _efficientnet, EfficientNet_V2_S_Weights
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def my_efficientnet_v2_s():
    """
    copy of efficientnet_v2_s in torchvision.model.efficientnet with change norm_layer=FrozenBatchNorm2d
    """
    weights = EfficientNet_V2_S_Weights.verify("DEFAULT")
    inverted_residual_setting, last_channel = _efficientnet_conf(
        "efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        0.2,
        last_channel,
        weights,
        progress=True,
        norm_layer=FrozenBatchNorm2d,
    )


# backbone = my_efficientnet_v2_s()
name = "resnet50"
# backbone = getattr(torchvision.models, name)(pretrained=True)
# body = backbone.features
# body = torch.nn.Sequential(*(list(body.children())[:-1]))
return_layers = {'layer4': "0"}
train_backbone = False
# print("train_backbone", train_backbone)
# for i in x:
#     print(x)
# for name, parameter in backbone.named_parameters():
#     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#         print(name)
# print(body)
# summary(body, input_size=(1, 3, 450, 613), depth=100)
from torchvision.ops import _new_empty_tensor
