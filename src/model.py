"""
src/model.py
Transfer learning architecture for RBC morphology classification.
"""

from typing import Optional
import torch
import torch.nn as nn
import timm


class SmearScopeClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone_name: str = "convnext_nano.in12k_ft_in1k_224",
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_model(
    num_classes: int,
    backbone_name: str = "convnext_nano.in12k_ft_in1k_224",
    pretrained: bool = True,
    device: Optional[str] = None
) -> SmearScopeClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SmearScopeClassifier(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained
    )
    return model.to(device)
