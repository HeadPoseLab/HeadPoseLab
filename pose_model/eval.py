import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pose_model.datasets.multimodal_sequence_dataset import (
    MultiPoseSequenceDataset,
    HEAD_NUM_CLASSES,
    HAND_NUM_CLASSES,
)
from pose_model.models.model_factory import build_pose_model, model_requires_images
from pose_model.utils.logger import get_logger
from pose_model.utils.losses import FocalLoss
from pose_model.utils.metrics import confusion, per_class_accuracy, sequence_accuracy, sequence_f1
from pose_model.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate head/hand pose model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--hand_unknown_threshold",
        type=float,
        default=None,
        help="If set, map low-confidence hand predictions to unknown class",
    )
    parser.add_argument(
        "--hand_unknown_class",
        type=int,
        default=None,
        help="1-based hand class id to use as unknown (default: last class)",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(preference: str):
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(counts, loss_cfg, device):
    mode = loss_cfg.get("class_weights", "none")
    if isinstance(mode, list):
        return torch.tensor(mode, dtype=torch.float, device=device)
    if isinstance(mode, str):
        mode_lower = mode.lower()
        if mode_lower == "none":
            return None
        if mode_lower == "auto":
            counts_tensor = torch.tensor(counts, dtype=torch.float, device=device).clamp_min(1.0)
            return counts_tensor.sum() / (len(counts_tensor) * counts_tensor)
    return None


def _apply_unknown_threshold(logits: torch.Tensor, threshold: float, unknown_index: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    max_probs, preds = probs.max(dim=-1)
    preds = preds.clone()
    preds[max_probs < threshold] = unknown_index
    return preds


def _metrics_from_preds(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    acc = (preds == labels).float().mean().item()
    y_pred = preds.detach().cpu().flatten().numpy()
    y_true = labels.detach().cpu().flatten().numpy()
    f1 = f1_score(y_true, y_pred, average="macro")
    cls_acc = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            cls_acc.append((preds[mask] == labels[mask]).float().mean().item())
        else:
            cls_acc.append(float("nan"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, f1, cls_acc, cm


def evaluate(
    model,
    loader,
    criterion_head,
    criterion_hand,
    device,
    head_weight: float,
    hand_weight: float,
    hand_unknown_threshold: float,
    hand_unknown_index: int,
):
    model.eval()
    total_loss = 0.0
    head_accs, hand_accs = [], []
    head_f1s, hand_f1s = [], []
    head_logits_all, hand_logits_all = [], []
    head_labels_all, hand_labels_all = [], []
    hand_preds_all = []
    with torch.no_grad():
        for head_images, hand_images, head_labels, hand_labels, head_coords, hand_coords in loader:
            head_images = head_images.to(device)
            hand_images = hand_images.to(device)
            head_labels = head_labels.to(device)
            hand_labels = hand_labels.to(device)
            head_coords = head_coords.to(device)
            hand_coords = hand_coords.to(device)
            head_logits, hand_logits = model(head_images, hand_images, head_coords, hand_coords)
            head_loss = criterion_head(head_logits.view(-1, HEAD_NUM_CLASSES), head_labels.view(-1))
            hand_loss = criterion_hand(hand_logits.view(-1, HAND_NUM_CLASSES), hand_labels.view(-1))
            loss = head_weight * head_loss + hand_weight * hand_loss
            total_loss += loss.item()
            head_accs.append(sequence_accuracy(head_logits, head_labels))
            if hand_unknown_threshold > 0:
                hand_preds = _apply_unknown_threshold(hand_logits, hand_unknown_threshold, hand_unknown_index)
                hand_preds_all.append(hand_preds.cpu())
                hand_labels_all.append(hand_labels.cpu())
            else:
                hand_accs.append(sequence_accuracy(hand_logits, hand_labels))
                hand_f1s.append(sequence_f1(hand_logits, hand_labels))
                hand_logits_all.append(hand_logits.cpu())
                hand_labels_all.append(hand_labels.cpu())
            head_f1s.append(sequence_f1(head_logits, head_labels))
            head_logits_all.append(head_logits.cpu())
            head_labels_all.append(head_labels.cpu())

    avg_loss = total_loss / len(loader)
    mean_head_acc = sum(head_accs) / len(head_accs)
    mean_head_f1 = sum(head_f1s) / len(head_f1s)
    head_logits_cat = torch.cat(head_logits_all, dim=0)
    head_labels_cat = torch.cat(head_labels_all, dim=0)
    head_cls_acc = per_class_accuracy(head_logits_cat, head_labels_cat, HEAD_NUM_CLASSES)
    head_cm = confusion(head_logits_cat, head_labels_cat, HEAD_NUM_CLASSES)
    if hand_unknown_threshold > 0:
        hand_preds_cat = torch.cat(hand_preds_all, dim=0)
        hand_labels_cat = torch.cat(hand_labels_all, dim=0)
        mean_hand_acc, mean_hand_f1, hand_cls_acc, hand_cm = _metrics_from_preds(
            hand_preds_cat, hand_labels_cat, HAND_NUM_CLASSES
        )
    else:
        mean_hand_acc = sum(hand_accs) / len(hand_accs)
        mean_hand_f1 = sum(hand_f1s) / len(hand_f1s)
        hand_logits_cat = torch.cat(hand_logits_all, dim=0)
        hand_labels_cat = torch.cat(hand_labels_all, dim=0)
        hand_cls_acc = per_class_accuracy(hand_logits_cat, hand_labels_cat, HAND_NUM_CLASSES)
        hand_cm = confusion(hand_logits_cat, hand_labels_cat, HAND_NUM_CLASSES)
    return avg_loss, mean_head_acc, mean_hand_acc, mean_head_f1, mean_hand_f1, head_cls_acc, hand_cls_acc, head_cm, hand_cm


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    eval_cfg = cfg.get("eval", {})
    hand_unknown_threshold = args.hand_unknown_threshold
    if hand_unknown_threshold is None:
        hand_unknown_threshold = float(eval_cfg.get("hand_unknown_threshold", 0.0))
    hand_unknown_class = args.hand_unknown_class
    if hand_unknown_class is None:
        hand_unknown_class = int(eval_cfg.get("hand_unknown_class", HAND_NUM_CLASSES))
    hand_unknown_index = max(0, min(hand_unknown_class - 1, HAND_NUM_CLASSES - 1))

    checkpoint_path = args.checkpoint or cfg["eval"]["checkpoint"]
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sampler_cfg = cfg.get("sampler", {})
    hand_roi_cfg = cfg.get("hand_roi", {})
    augment_cfg = cfg.get("augment", {})
    load_images = model_requires_images(cfg)
    dataset = MultiPoseSequenceDataset(
        data_root=cfg["data_root"],
        mode="test",
        sequence_length=cfg["sequence_length"],
        overlap=cfg["overlap"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        image_size=cfg["image_size"],
        head_dir=cfg.get("head_dir", "head_pose"),
        hand_dir=cfg.get("hand_dir", "hand_pose"),
        sample_weight_head=sampler_cfg.get("sample_weight_head", 0.5),
        sample_weight_hand=sampler_cfg.get("sample_weight_hand", 0.5),
        hand_roi_enabled=hand_roi_cfg.get("enabled", False),
        hand_roi_expand=hand_roi_cfg.get("expand", 1.6),
        hand_roi_min_scale=hand_roi_cfg.get("min_scale", 0.2),
        augment_cfg=augment_cfg,
        load_images=load_images,
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model, model_arch = build_pose_model(cfg, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    logger.info("Model architecture: %s", model_arch)
    if hand_unknown_threshold and hand_unknown_threshold > 0:
        logger.info(
            "Applying hand unknown threshold %.3f -> class %d",
            hand_unknown_threshold,
            hand_unknown_class,
        )

    head_counts = [dataset.class_counts_head.get(i, 0) for i in range(HEAD_NUM_CLASSES)]
    hand_counts = [dataset.class_counts_hand.get(i, 0) for i in range(HAND_NUM_CLASSES)]
    class_weights_head = compute_class_weights(head_counts, cfg["loss"], device)
    class_weights_hand = compute_class_weights(hand_counts, cfg["loss"], device)
    head_weight = cfg["loss"].get("head_weight", 1.0)
    hand_weight = cfg["loss"].get("hand_weight", 1.0)
    loss_cfg = cfg["loss"]
    head_loss_type = loss_cfg.get("head_type", loss_cfg.get("type", "cross_entropy"))
    hand_loss_type = loss_cfg.get("hand_type", loss_cfg.get("type", "cross_entropy"))
    head_gamma = loss_cfg.get("head_focal_gamma", loss_cfg.get("focal_gamma", 2.0))
    hand_gamma = loss_cfg.get("hand_focal_gamma", loss_cfg.get("focal_gamma", 2.0))
    if head_loss_type == "focal":
        criterion_head = FocalLoss(gamma=head_gamma, weight=class_weights_head)
    else:
        criterion_head = nn.CrossEntropyLoss(weight=class_weights_head)
    if hand_loss_type == "focal":
        criterion_hand = FocalLoss(gamma=hand_gamma, weight=class_weights_hand)
    else:
        criterion_hand = nn.CrossEntropyLoss(weight=class_weights_hand)
    (
        loss,
        head_acc,
        hand_acc,
        head_f1,
        hand_f1,
        head_cls_acc,
        hand_cls_acc,
        head_cm,
        hand_cm,
    ) = evaluate(
        model,
        loader,
        criterion_head,
        criterion_hand,
        device,
        head_weight,
        hand_weight,
        hand_unknown_threshold,
        hand_unknown_index,
    )
    logger.info("Test loss=%.4f head_acc=%.4f hand_acc=%.4f head_f1=%.4f hand_f1=%.4f", loss, head_acc, hand_acc, head_f1, hand_f1)
    logger.info(
        "Head per-class accuracy (0-%d): %s",
        HEAD_NUM_CLASSES - 1,
        ["{:.3f}".format(x) for x in head_cls_acc],
    )
    logger.info(
        "Hand per-class accuracy (0-%d): %s",
        HAND_NUM_CLASSES - 1,
        ["{:.3f}".format(x) for x in hand_cls_acc],
    )
    logger.info("Head confusion matrix:\n%s", head_cm)
    logger.info("Hand confusion matrix:\n%s", hand_cm)


if __name__ == "__main__":
    main()
