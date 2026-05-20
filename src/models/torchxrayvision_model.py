import torch
import torch.nn.functional as F
from torch import nn

import torchxrayvision as xrv


DEFAULT_WEIGHTS = "densenet121-res224-chex"
FORBIDDEN_WEIGHT_TOKENS = ("rsna", "all")


def validate_torchxrayvision_weights(weights):
    weight_name = weights.lower()
    if any(token in weight_name for token in FORBIDDEN_WEIGHT_TOKENS):
        raise ValueError(
            f"Forbidden TorchXRayVision weights: {weights}. "
            "Use weights that do not include RSNA or all."
        )
    return weights


class TorchXRayVisionDenseNetBinary(nn.Module):
    def __init__(self, weights=DEFAULT_WEIGHTS):
        super().__init__()
        weights = validate_torchxrayvision_weights(weights)
        self.weights = weights
        self.backbone = xrv.models.DenseNet(weights=weights)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Linear(in_features, 1)

        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prepare_input(self, x):
        if x.shape[1] == 1:
            return x
        if x.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got {x.shape[1]}.")

        # KaggleChestXrayDataset returns ImageNet-normalized RGB tensors.
        x = (x * self.imagenet_std + self.imagenet_mean).clamp(0.0, 1.0)
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        return x * 2048.0 - 1024.0

    def forward(self, x):
        x = self._prepare_input(x)
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        logits = self.head(features)
        return logits.squeeze(1)
