from typing import Optional

import torch
from torch import nn
from torchvision import models


class SimpleCNNBackbone(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, feature_dim: int = 256, freeze_backbone: bool = False, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features
        self.proj = nn.Linear(in_features, feature_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


def create_backbone(name: str, feature_dim: int, freeze_backbone: bool = False):
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNNBackbone(feature_dim=feature_dim)
    if name == "resnet18":
        return ResNet18Backbone(feature_dim=feature_dim, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unknown backbone: {name}")
