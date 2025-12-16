from typing import List, Sequence, Tuple

import torch
from torch import nn

from .cnn_backbone import create_backbone

NUM_CLASSES = 5


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dilations: Sequence[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(len(channels))]
        if len(dilations) != len(channels):
            raise ValueError("dilations and channels must have same length")

        layers: List[nn.Module] = []
        in_ch = input_dim
        for out_ch, dil in zip(channels, dilations):
            layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dil, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.output_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.network(x)
        return x.transpose(1, 2)  # [B, T, D_out]


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        backbone: str = "simple_cnn",
        feature_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
        temporal_encoder: str = "lstm",
        tcn_channels: Sequence[int] | None = None,
        tcn_kernel: int = 3,
        tcn_dilations: Sequence[int] | None = None,
        tcn_dropout: float = 0.2,
        freeze_backbone: bool = False,
        freeze_stages: int = -1,
        pretrained: bool = True,
        resnet_variant: str = "resnet18",
        cnn_branch_channels=None,
        fusion: str = "concat",
        fusion_dropout: float = 0.0,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.backbone = create_backbone(
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
        encoder = temporal_encoder.lower()
        self.temporal_encoder = encoder

        if encoder == "tcn":
            channels = list(tcn_channels) if tcn_channels is not None else [feature_dim, feature_dim]
            self.tcn = TemporalConvNet(
                input_dim=feature_dim,
                channels=channels,
                kernel_size=tcn_kernel,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
            )
            self.classifier = nn.Linear(self.tcn.output_dim, num_classes)
            self.lstm = None
        elif encoder == "lstm":
            self.tcn = None
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            num_dirs = 2 if bidirectional else 1
            self.classifier = nn.Linear(lstm_hidden * num_dirs, num_classes)
        else:
            raise ValueError(f"Unsupported temporal_encoder: {temporal_encoder}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, C, H, W]
        b, t, c, h, w = x.shape
        features = self.backbone(x.view(b * t, c, h, w))
        features = features.view(b, t, -1)
        if self.temporal_encoder == "tcn":
            encoded = self.tcn(features)
        else:
            encoded, _ = self.lstm(features)
        logits = self.classifier(encoded)
        return logits, encoded
