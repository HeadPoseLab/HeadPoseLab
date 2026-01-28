from typing import Sequence

import torch
from torch import nn

from .cnn_backbone import create_backbone
from .cnn_lstm import TemporalConvNet
from .transformer_encoder import TransformerEncoder


class TemporalAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)
        context = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return self.dropout(context)


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


class BranchAdapter(nn.Module):
    def __init__(self, feature_dim: int, adapter_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = adapter_dim or feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual adapter on [B, T, D]
        return x + self.net(x)


class MultiTaskPoseModel(nn.Module):
    def __init__(
        self,
        backbone: str = "simple_cnn",
        feature_dim: int = 128,
        temporal_encoder: str = "transformer",
        head_temporal_encoder: str | None = None,
        hand_temporal_encoder: str | None = None,
        tcn_channels: Sequence[int] | None = None,
        tcn_kernel: int = 3,
        tcn_dilations: Sequence[int] | None = None,
        tcn_dropout: float = 0.2,
        head_tcn_channels: Sequence[int] | None = None,
        hand_tcn_channels: Sequence[int] | None = None,
        head_tcn_kernel: int | None = None,
        hand_tcn_kernel: int | None = None,
        head_tcn_dilations: Sequence[int] | None = None,
        hand_tcn_dilations: Sequence[int] | None = None,
        head_tcn_dropout: float | None = None,
        hand_tcn_dropout: float | None = None,
        transformer_cfg: dict | None = None,
        head_transformer_cfg: dict | None = None,
        hand_transformer_cfg: dict | None = None,
        shared_backbone: bool = False,
        shared_temporal: bool = False,
        adapter_enabled: bool = False,
        adapter_dim: int | None = None,
        adapter_dropout: float = 0.1,
        hand_use_attn_pool: bool = False,
        hand_attn_pool_dropout: float = 0.1,
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

        self.head_adapter = BranchAdapter(feature_dim, adapter_dim, adapter_dropout) if adapter_enabled else nn.Identity()
        self.hand_adapter = BranchAdapter(feature_dim, adapter_dim, adapter_dropout) if adapter_enabled else nn.Identity()

        head_encoder = (head_temporal_encoder or temporal_encoder).lower()
        hand_encoder = (hand_temporal_encoder or temporal_encoder).lower()

        base_transformer_cfg = transformer_cfg or {}
        head_transformer_cfg = head_transformer_cfg or base_transformer_cfg
        hand_transformer_cfg = hand_transformer_cfg or base_transformer_cfg

        head_tcn_channels = head_tcn_channels or tcn_channels
        hand_tcn_channels = hand_tcn_channels or tcn_channels
        head_tcn_kernel = head_tcn_kernel or tcn_kernel
        hand_tcn_kernel = hand_tcn_kernel or tcn_kernel
        head_tcn_dilations = head_tcn_dilations or tcn_dilations
        hand_tcn_dilations = hand_tcn_dilations or tcn_dilations
        head_tcn_dropout = head_tcn_dropout if head_tcn_dropout is not None else tcn_dropout
        hand_tcn_dropout = hand_tcn_dropout if hand_tcn_dropout is not None else tcn_dropout

        if shared_temporal and head_encoder == hand_encoder:
            temporal_head = self._build_temporal(
                head_encoder,
                feature_dim,
                head_tcn_channels,
                head_tcn_kernel,
                head_tcn_dilations,
                head_tcn_dropout,
                head_transformer_cfg,
            )
            self.temporal_head = temporal_head
            self.temporal_hand = temporal_head
        else:
            self.temporal_head = self._build_temporal(
                head_encoder,
                feature_dim,
                head_tcn_channels,
                head_tcn_kernel,
                head_tcn_dilations,
                head_tcn_dropout,
                head_transformer_cfg,
            )
            self.temporal_hand = self._build_temporal(
                hand_encoder,
                feature_dim,
                hand_tcn_channels,
                hand_tcn_kernel,
                hand_tcn_dilations,
                hand_tcn_dropout,
                hand_transformer_cfg,
            )

        self.hand_use_attn_pool = bool(hand_use_attn_pool)
        self.hand_attn_pool = (
            TemporalAttentionPooling(self.temporal_hand.output_dim, hand_attn_pool_dropout)
            if self.hand_use_attn_pool
            else None
        )

        self.head_classifier = nn.Linear(self.temporal_head.output_dim, num_head_classes)
        self.hand_classifier = nn.Linear(self.temporal_hand.output_dim, num_hand_classes)

    @staticmethod
    def _build_temporal(
        encoder: str,
        feature_dim: int,
        tcn_channels: Sequence[int] | None,
        tcn_kernel: int,
        tcn_dilations: Sequence[int] | None,
        tcn_dropout: float,
        transformer_cfg: dict,
    ) -> nn.Module:
        if encoder == "tcn":
            return TCNTemporal(feature_dim, tcn_channels, tcn_kernel, tcn_dilations, tcn_dropout)
        if encoder == "transformer":
            return TransformerTemporal(feature_dim, transformer_cfg)
        return TransformerTemporal(feature_dim, transformer_cfg)

    @staticmethod
    def _extract_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        feats = backbone(x.view(b * t, c, h, w))
        return feats.view(b, t, -1)

    def forward(self, head_images: torch.Tensor, hand_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head_features = self._extract_features(self.backbone_head, head_images)
        hand_features = self._extract_features(self.backbone_hand, hand_images)

        head_features = self.head_adapter(head_features)
        hand_features = self.hand_adapter(hand_features)

        head_encoded = self.temporal_head(head_features)
        hand_encoded = self.temporal_hand(hand_features)
        if self.hand_attn_pool is not None:
            hand_context = self.hand_attn_pool(hand_encoded)
            hand_encoded = hand_encoded + hand_context.unsqueeze(1)

        head_logits = self.head_classifier(head_encoded)
        hand_logits = self.hand_classifier(hand_encoded)
        return head_logits, hand_logits
