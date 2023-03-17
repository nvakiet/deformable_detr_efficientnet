from torchvision.models.mobilenet import mobilenet_v3_large
from torchvision.models._utils import IntermediateLayerGetter
from torchinfo import summary
mobilenet_model = mobilenet_v3_large(weights="DEFAULT")
# print(mobilenet_model)

body = IntermediateLayerGetter(
    mobilenet_model, return_layers={"features": "0"})

summary(body, input_size=(1, 3, 450, 613), depth=100)
