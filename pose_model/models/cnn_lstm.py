from typing import Tuple

import torch
from torch import nn

from .cnn_backbone import create_backbone

NUM_CLASSES = 7


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        backbone: str = "simple_cnn",
        feature_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone, feature_dim, freeze_backbone=freeze_backbone)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq, C, H, W]
        b, t, c, h, w = x.shape
        features = self.backbone(x.view(b * t, c, h, w))
        features = features.view(b, t, -1)
        lstm_out, _ = self.lstm(features)
        logits = self.classifier(lstm_out)
        return logits, lstm_out
