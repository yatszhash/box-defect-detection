import torchvision
from torch import nn


def create_pretrained_resnet50(dropout_rate=0.4, inner_units=32, n_class=1):
    resnet = torchvision.models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.required_grad = False
    out_channel = resnet.fc.out_features
    return nn.Sequential(
        resnet,
        nn.Dropout(dropout_rate),
        nn.Linear(out_channel, inner_units),
        nn.Dropout(dropout_rate),
        nn.Linear(inner_units, n_class)
    )
