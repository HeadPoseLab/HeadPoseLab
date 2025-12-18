from typing import Iterable, Optional, Sequence

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


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        variant: str = "resnet50",
        freeze_backbone: bool = False,
        freeze_stages: int = -1,
        pretrained: bool = True,
    ):
        super().__init__()
        variant = variant.lower()
        if variant == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported resnet variant: {variant}. Only resnet50 is supported.")

        self.stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        in_features = base_model.fc.in_features
        self.proj = nn.Linear(in_features, feature_dim)

        effective_freeze = 4 if freeze_backbone and freeze_stages < 0 else freeze_stages
        self._freeze_stages(effective_freeze)

    @staticmethod
    def _freeze_module(module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def _freeze_stages(self, freeze_stages: int):
        # -1 means no freeze; 0 freezes stem; 1 freezes stem+layer1; ...; 4 freezes all backbone layers.
        if freeze_stages < 0:
            return
        if freeze_stages >= 0:
            self._freeze_module(self.stem)
        if freeze_stages >= 1:
            self._freeze_module(self.layer1)
        if freeze_stages >= 2:
            self._freeze_module(self.layer2)
        if freeze_stages >= 3:
            self._freeze_module(self.layer3)
        if freeze_stages >= 4:
            self._freeze_module(self.layer4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class HybridCNNResNetBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        cnn_branch_channels: Sequence[int] = (32, 64, 128),
        resnet_variant: str = "resnet50",
        freeze_backbone: bool = False,
        freeze_stages: int = -1,
        pretrained: bool = True,
        fusion: str = "concat",
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        fusion = fusion.lower()
        if fusion not in {"concat", "sum"}:
            raise ValueError(f"Unsupported fusion mode: {fusion}")
        # CNN branch
        cnn_layers = []
        in_ch = 3
        for ch in cnn_branch_channels:
            cnn_layers.extend(
                [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            )
            in_ch = ch
        self.cnn_branch = nn.Sequential(*cnn_layers)
        self.cnn_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet branch
        res_feat_dim = max(feature_dim // (2 if fusion == "concat" else 1), 1)
        cnn_feat_dim = res_feat_dim if fusion == "sum" else res_feat_dim
        self.resnet_branch = ResNetBackbone(
            feature_dim=res_feat_dim,
            variant=resnet_variant,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
        self.cnn_proj = nn.Linear(cnn_branch_channels[-1], cnn_feat_dim)

        fused_dim = cnn_feat_dim + res_feat_dim if fusion == "concat" else res_feat_dim
        self.fusion = fusion
        self.fusion_norm = nn.LayerNorm(fused_dim)
        self.fusion_dropout = nn.Dropout(fusion_dropout) if fusion_dropout and fusion_dropout > 0 else nn.Identity()
        self.fused_proj = nn.Linear(fused_dim, feature_dim)

    @staticmethod
    def _flatten_gap(tensor: torch.Tensor) -> torch.Tensor:
        return torch.flatten(tensor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN path
        cnn_feat = self.cnn_branch(x)
        cnn_feat = self.cnn_gap(cnn_feat)
        cnn_feat = self._flatten_gap(cnn_feat)
        cnn_feat = self.cnn_proj(cnn_feat)

        # ResNet path
        res_feat = self.resnet_branch(x)

        if self.fusion == "concat":
            fused = torch.cat([cnn_feat, res_feat], dim=1)
        else:
            fused = cnn_feat + res_feat

        fused = self.fusion_norm(fused)
        fused = self.fusion_dropout(fused)
        return self.fused_proj(fused)


def create_backbone(
    name: str,
    feature_dim: int,
    freeze_backbone: bool = False,
    freeze_stages: int = -1,
    pretrained: bool = True,
    resnet_variant: str = "resnet50",
    cnn_branch_channels: Optional[Iterable[int]] = None,
    fusion: str = "concat",
    fusion_dropout: float = 0.0,
):
    name = name.lower()
    if name == "simple_cnn":
        return SimpleCNNBackbone(feature_dim=feature_dim)
    if name == "resnet50":
        return ResNetBackbone(
            feature_dim=feature_dim,
            variant="resnet50",
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
        )
    if name == "hybrid_cnn_resnet":
        channels = tuple(cnn_branch_channels) if cnn_branch_channels is not None else (32, 64, 128)
        return HybridCNNResNetBackbone(
            feature_dim=feature_dim,
            cnn_branch_channels=channels,
            resnet_variant=resnet_variant,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
            fusion=fusion,
            fusion_dropout=fusion_dropout,
        )
    raise ValueError(f"Unknown backbone: {name}")
