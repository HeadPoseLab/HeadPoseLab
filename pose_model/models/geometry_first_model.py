from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .cnn_backbone import create_backbone


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


class TemporalGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return out


class GeometryFeatureBuilder(nn.Module):
    HEAD_IN_DIM = 19
    HAND_IN_DIM = 23

    @staticmethod
    def _point_mask(point_xy: torch.Tensor) -> torch.Tensor:
        # point_xy: [B, T, 2]
        return ((point_xy[..., 0] > 0) | (point_xy[..., 1] > 0)).float().unsqueeze(-1)

    @staticmethod
    def _temporal_diff(x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        first = torch.zeros_like(x[:, :1])
        return torch.cat([first, x[:, 1:] - x[:, :-1]], dim=1)

    def forward(self, head_coords: torch.Tensor, hand_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # head_coords: [B, T, 2], hand_coords: [B, T, 4]
        head_xy = head_coords[..., :2]
        left_xy = hand_coords[..., :2]
        right_xy = hand_coords[..., 2:4]

        head_mask = self._point_mask(head_xy)
        left_mask = self._point_mask(left_xy)
        right_mask = self._point_mask(right_xy)
        hand_presence = ((left_mask + right_mask) > 0).float()

        denom = (left_mask + right_mask).clamp_min(1.0)
        hand_center = (left_xy * left_mask + right_xy * right_mask) / denom
        hand_span = torch.norm(left_xy - right_xy, dim=-1, keepdim=True)

        rel_head_hand = head_xy - hand_center
        rel_left_head = left_xy - head_xy
        rel_right_head = right_xy - head_xy

        head_vel = self._temporal_diff(head_xy)
        hand_center_vel = self._temporal_diff(hand_center)
        left_vel = self._temporal_diff(left_xy)
        right_vel = self._temporal_diff(right_xy)

        mask_feats = torch.cat([head_mask, left_mask, right_mask, hand_presence], dim=-1)

        head_features = torch.cat(
            [
                head_xy,
                hand_center,
                rel_head_hand,
                hand_span,
                head_vel,
                hand_center_vel,
                left_xy,
                right_xy,
                mask_feats,
            ],
            dim=-1,
        )
        hand_features = torch.cat(
            [
                left_xy,
                right_xy,
                hand_center,
                rel_left_head,
                rel_right_head,
                hand_span,
                left_vel,
                right_vel,
                head_xy,
                head_vel,
                mask_feats,
            ],
            dim=-1,
        )
        return head_features, hand_features


class GeometryFirstPoseModel(nn.Module):
    def __init__(
        self,
        geom_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        num_head_classes: int = 5,
        num_hand_classes: int = 4,
        use_head_attn_pool: bool = True,
        use_hand_attn_pool: bool = True,
        attn_pool_dropout: float = 0.1,
        appearance_expert_enabled: bool = False,
        appearance_backbone: str = "simple_cnn",
        appearance_feature_dim: int = 96,
        appearance_shared_backbone: bool = True,
        appearance_dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_builder = GeometryFeatureBuilder()
        self.head_proj = nn.Sequential(
            nn.Linear(self.feature_builder.HEAD_IN_DIM, geom_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(geom_dim),
        )
        self.hand_proj = nn.Sequential(
            nn.Linear(self.feature_builder.HAND_IN_DIM, geom_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(geom_dim),
        )

        self.temporal_head = TemporalGRUEncoder(
            input_dim=geom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.temporal_hand = TemporalGRUEncoder(
            input_dim=geom_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        temporal_dim = self.temporal_head.output_dim
        self.head_attn_pool = (
            TemporalAttentionPooling(temporal_dim, attn_pool_dropout) if use_head_attn_pool else None
        )
        self.hand_attn_pool = (
            TemporalAttentionPooling(temporal_dim, attn_pool_dropout) if use_hand_attn_pool else None
        )

        self.head_classifier = nn.Linear(temporal_dim, num_head_classes)
        self.hand_classifier = nn.Linear(temporal_dim, num_hand_classes)

        self.appearance_expert_enabled = bool(appearance_expert_enabled)
        if self.appearance_expert_enabled:
            self.appearance_backbone_head = create_backbone(
                appearance_backbone,
                appearance_feature_dim,
                pretrained=True,
            )
            self.appearance_backbone_hand = (
                self.appearance_backbone_head
                if appearance_shared_backbone
                else create_backbone(
                    appearance_backbone,
                    appearance_feature_dim,
                    pretrained=True,
                )
            )
            self.appearance_dropout = nn.Dropout(appearance_dropout) if appearance_dropout > 0 else nn.Identity()
            self.head_appearance_classifier = nn.Linear(appearance_feature_dim, num_head_classes)
            self.hand_appearance_classifier = nn.Linear(appearance_feature_dim, num_hand_classes)
            self.head_gate = nn.Linear(temporal_dim, 1)
            self.hand_gate = nn.Linear(temporal_dim, 1)
        else:
            self.appearance_backbone_head = None
            self.appearance_backbone_hand = None
            self.appearance_dropout = nn.Identity()
            self.head_appearance_classifier = None
            self.hand_appearance_classifier = None
            self.head_gate = None
            self.hand_gate = None

    def get_backbone_modules(self) -> list[nn.Module]:
        if not self.appearance_expert_enabled:
            return []
        modules: list[nn.Module] = []
        for module in (self.appearance_backbone_head, self.appearance_backbone_hand):
            if module is not None and module not in modules:
                modules.append(module)
        return modules

    @staticmethod
    def _extract_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
        # images: [B, T, C, H, W]
        b, t, c, h, w = images.shape
        feats = backbone(images.view(b * t, c, h, w))
        return feats.view(b, t, -1)

    def forward(
        self,
        head_images: torch.Tensor | None,
        hand_images: torch.Tensor | None,
        head_coords: torch.Tensor | None = None,
        hand_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if head_coords is None or hand_coords is None:
            raise ValueError("GeometryFirstPoseModel requires head_coords and hand_coords.")

        head_feat, hand_feat = self.feature_builder(head_coords, hand_coords)
        head_feat = self.head_proj(head_feat)
        hand_feat = self.hand_proj(hand_feat)

        head_encoded = self.temporal_head(head_feat)
        hand_encoded = self.temporal_hand(hand_feat)

        if self.head_attn_pool is not None:
            head_context = self.head_attn_pool(head_encoded)
            head_encoded = head_encoded + head_context.unsqueeze(1)
        if self.hand_attn_pool is not None:
            hand_context = self.hand_attn_pool(hand_encoded)
            hand_encoded = hand_encoded + hand_context.unsqueeze(1)

        head_logits = self.head_classifier(head_encoded)
        hand_logits = self.hand_classifier(hand_encoded)

        if self.appearance_expert_enabled:
            if head_images is None or hand_images is None:
                raise ValueError("appearance_expert_enabled requires head_images and hand_images.")
            head_app_feat = self._extract_features(self.appearance_backbone_head, head_images)
            hand_app_feat = self._extract_features(self.appearance_backbone_hand, hand_images)
            head_app_logits = self.head_appearance_classifier(self.appearance_dropout(head_app_feat))
            hand_app_logits = self.hand_appearance_classifier(self.appearance_dropout(hand_app_feat))
            head_gate = torch.sigmoid(self.head_gate(head_encoded))
            hand_gate = torch.sigmoid(self.hand_gate(hand_encoded))
            head_logits = head_logits + head_gate * head_app_logits
            hand_logits = hand_logits + hand_gate * hand_app_logits

        return head_logits, hand_logits
