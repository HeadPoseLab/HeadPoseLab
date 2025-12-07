import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        focal_factor = (1 - probs) ** self.gamma
        loss = F.nll_loss(focal_factor * log_probs, targets, weight=self.weight, reduction=self.reduction)
        return loss
