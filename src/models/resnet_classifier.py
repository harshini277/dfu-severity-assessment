
import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.backbone = models.resnet18(weights=weights)

        feature_count = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            feature_count,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)
