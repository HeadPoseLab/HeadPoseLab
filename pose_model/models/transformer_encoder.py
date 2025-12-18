import math
from typing import Optional

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal or learned positional encoding that preserves input shape."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 500,
        learned: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.learned = learned

        if learned:
            pe = torch.zeros(1, max_len, d_model)
            nn.init.normal_(pe, mean=0.0, std=0.02)
            self.pe = nn.Parameter(pe)
        else:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        seq_len = x.size(1)
        if self.learned:
            x = x + self.pe[:, :seq_len]
        else:
            # For sinusoidal encoding, buffer is not a Parameter
            x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Wrapper around nn.TransformerEncoder with optional positional encoding."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        use_positional_encoding: bool = True,
        pos_encoding: str = "learned",
        pre_norm: bool = True,
        max_len: int = 500,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_norm,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pos_encoding = pos_encoding.lower()
        self.pos_encoder: Optional[nn.Module] = None
        if use_positional_encoding:
            learned = pos_encoding.startswith("learned")
            self.pos_encoder = PositionalEncoding(
                d_model=input_dim,
                dropout=dropout,
                max_len=max_len,
                learned=learned,
            )

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        if self.pos_encoder:
            x = self.pos_encoder(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
