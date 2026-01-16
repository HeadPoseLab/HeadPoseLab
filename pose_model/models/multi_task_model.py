from typing import Sequence

import torch
from torch import nn

from .cnn_backbone import create_backbone
from .cnn_lstm import TemporalConvNet
from .transformer_encoder import TransformerEncoder


class TransformerTemporal(nn.Module):
    def __init__(self, feature_dim: int, cfg: dict):
        super().__init__()
        transformer_dim = cfg.get("input_dim", feature_dim) or feature_dim
        self.input_proj = nn.Linear(feature_dim, transformer_dim) if transformer_dim != feature_dim else nn.Identity()
        self.encoder = TransformerEncoder(
            input_dim=transformer_dim,
            num_layers=cfg.get("num_layers", 2),
            nhead=cfg.get("nhead", 4),
            dim_feedforward=cfg.get("dim_feedforward", 512),
            dropout=cfg.get("dropout", 0.1),
            activation=cfg.get("activation", "relu"),
            use_positional_encoding=cfg.get("use_positional_encoding", True),
            pos_encoding=cfg.get("pos_encoding", "learned"),
            pre_norm=cfg.get("pre_norm", True),
            max_len=cfg.get("max_len", 500),
        )
        self.output_dim = transformer_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        return self.encoder(x)


class TCNTemporal(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        channels: Sequence[int] | None,
        kernel: int,
        dilations: Sequence[int] | None,
        dropout: float,
    ):
        super().__init__()
        channel_list = list(channels) if channels is not None else [feature_dim, feature_dim]
        self.tcn = TemporalConvNet(
            input_dim=feature_dim,
            channels=channel_list,
            kernel_size=kernel,
            dilations=dilations,
            dropout=dropout,
        )
        self.output_dim = self.tcn.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)


class MultiTaskPoseModel(nn.Module):
    def __init__(
        self,
        backbone: str = "simple_cnn",
        feature_dim: int = 128,
        temporal_encoder: str = "transformer",
        tcn_channels: Sequence[int] | None = None,
        tcn_kernel: int = 3,
        tcn_dilations: Sequence[int] | None = None,
        tcn_dropout: float = 0.2,
        transformer_cfg: dict | None = None,
        shared_backbone: bool = False,
        shared_temporal: bool = False,
        num_head_classes: int = 5,
        num_hand_classes: int = 4,
        freeze_backbone: bool = False,
        freeze_stages: int = -1,
        pretrained: bool = True,
        resnet_variant: str = "resnet50",
        cnn_branch_channels=None,
        fusion: str = "concat",
        fusion_dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal_encoder = temporal_encoder.lower()

        self.backbone_head = create_backbone(
            backbone,
            feature_dim,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
            resnet_variant=resnet_variant,
            cnn_branch_channels=cnn_branch_channels,
            fusion=fusion,
            fusion_dropout=fusion_dropout,
        )
        self.backbone_hand = self.backbone_head if shared_backbone else create_backbone(
            backbone,
            feature_dim,
            freeze_backbone=freeze_backbone,
            freeze_stages=freeze_stages,
            pretrained=pretrained,
            resnet_variant=resnet_variant,
            cnn_branch_channels=cnn_branch_channels,
            fusion=fusion,
            fusion_dropout=fusion_dropout,
        )

        transformer_cfg = transformer_cfg or {}
        if self.temporal_encoder == "tcn":
            temporal_head = TCNTemporal(feature_dim, tcn_channels, tcn_kernel, tcn_dilations, tcn_dropout)
        elif self.temporal_encoder == "transformer":
            temporal_head = TransformerTemporal(feature_dim, transformer_cfg)
        else:
            raise ValueError(f"Unsupported temporal_encoder: {temporal_encoder} (only tcn or transformer supported)")

        self.temporal_head = temporal_head
        self.temporal_hand = temporal_head if shared_temporal else self._clone_temporal(
            temporal_head, feature_dim, tcn_channels, tcn_kernel, tcn_dilations, tcn_dropout, transformer_cfg
        )

        self.head_classifier = nn.Linear(self.temporal_head.output_dim, num_head_classes)
        self.hand_classifier = nn.Linear(self.temporal_hand.output_dim, num_hand_classes)

    def _clone_temporal(
        self,
        base_temporal: nn.Module,
        feature_dim: int,
        tcn_channels: Sequence[int] | None,
        tcn_kernel: int,
        tcn_dilations: Sequence[int] | None,
        tcn_dropout: float,
        transformer_cfg: dict,
    ) -> nn.Module:
        if isinstance(base_temporal, TCNTemporal):
            return TCNTemporal(feature_dim, tcn_channels, tcn_kernel, tcn_dilations, tcn_dropout)
        return TransformerTemporal(feature_dim, transformer_cfg)

    @staticmethod
    def _extract_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        feats = backbone(x.view(b * t, c, h, w))
        return feats.view(b, t, -1)

    def forward(self, head_images: torch.Tensor, hand_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_features = self._extract_features(self.backbone_head, head_images)
        hand_features = self._extract_features(self.backbone_hand, hand_images)

        head_encoded = self.temporal_head(head_features)
        hand_encoded = self.temporal_hand(hand_features)

        head_logits = self.head_classifier(head_encoded)
        hand_logits = self.hand_classifier(hand_encoded)
        return head_logits, hand_logits
