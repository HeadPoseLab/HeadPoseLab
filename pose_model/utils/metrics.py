from typing import List

import torch
from sklearn.metrics import confusion_matrix, f1_score


def sequence_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float().mean()
    return correct.item()


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> List[float]:
    preds = logits.argmax(dim=-1)
    results = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            cls_acc = (preds[mask] == labels[mask]).float().mean().item()
        else:
            cls_acc = float("nan")
        results.append(cls_acc)
    return results


def sequence_f1(logits: torch.Tensor, labels: torch.Tensor, average: str = "macro") -> float:
    y_pred = logits.argmax(dim=-1).detach().cpu().flatten().numpy()
    y_true = labels.detach().cpu().flatten().numpy()
    return f1_score(y_true, y_pred, average=average)


def confusion(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    y_pred = logits.argmax(dim=-1).detach().cpu().flatten().numpy()
    y_true = labels.detach().cpu().flatten().numpy()
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
