
import torch
import torch.nn as nn
from torchvision import models



class CoralResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        weights = (
            models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.backbone = models.resnet18(weights=weights)

        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.feature_layer = nn.Linear(
            feature_count, 1, bias=False
        )

        # Positive increments ensure thresholds stay ordered.
        self.raw_threshold_deltas = nn.Parameter(
            torch.zeros(num_classes - 1)
        )

    @property
    def thresholds(self):
        return torch.cumsum(
            torch.nn.functional.softplus(
                self.raw_threshold_deltas
            ),
            dim=0
        )

    def forward(self, x):
        features = self.backbone(x)
        scores = self.feature_layer(features)

        return scores - self.thresholds.unsqueeze(0)



def coral_targets(labels, num_classes=4):
    """
    Convert class labels into cumulative ordinal targets.

    Example:
        Grade 1 (0) -> [0, 0, 0]
        Grade 2 (1) -> [1, 0, 0]
        Grade 3 (2) -> [1, 1, 0]
        Grade 4 (3) -> [1, 1, 1]
    """
    thresholds = torch.arange(
        num_classes - 1,
        device=labels.device
    )

    return (labels.unsqueeze(1) > thresholds).float()


def coral_loss(logits, labels, num_classes=4):
    """
    Binary cross-entropy loss over cumulative ordinal targets.
    """
    targets = coral_targets(labels, num_classes)

    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )


def coral_predict(logits):
    """
    Convert ordinal logits into predicted class indices.
    """
    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > 0.5).sum(dim=1)

    return predictions
