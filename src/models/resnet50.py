import torch
from torch import nn

import timm


class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=1,
        )

    def forward(self, x):
        logits = self.model(x)
        return logits.squeeze(1)
