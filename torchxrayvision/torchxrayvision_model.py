import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchxrayvision as xrv
except ImportError as exc:
    raise ImportError(
        "torchxrayvision package is not installed or not visible. "
        "Check PYTHONPATH or install torchxrayvision."
    ) from exc


DEFAULT_WEIGHTS = "densenet121-res224-all"


class TorchXRayVisionDenseNetBinary(nn.Module):
    """
    TorchXRayVision DenseNet121 backbone + binary pneumonia classifier.

    Important:
    TorchXRayVision DenseNet forward() applies op_norm using 18 pathology thresholds.
    If we replace its classifier with a 1-output binary head and still call
    self.backbone(x), op_norm crashes because it expects 18 outputs.

    Therefore, this wrapper uses the DenseNet feature extractor directly and
    applies our own binary classifier head.
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, dropout=0.2):
        super().__init__()

        self.weights = weights
        self.backbone = xrv.models.DenseNet(weights=weights)

        in_features = self.backbone.classifier.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        # Dataset currently returns RGB 3-channel images.
        # TorchXRayVision models are trained on single-channel X-rays.
        if x.dim() == 4 and x.size(1) == 3:
            x = x.mean(dim=1, keepdim=True)

        # TorchXRayVision expects X-ray-like values, but we use the current
        # project preprocessing. This avoids calling backbone.forward(),
        # because backbone.forward() applies 18-output op_norm.
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        logits = self.classifier(pooled)

        return logits.reshape(-1)