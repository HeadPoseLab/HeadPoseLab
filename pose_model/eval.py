import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pose_model.datasets.sequence_dataset import PoseSequenceDataset, NUM_CLASSES
from pose_model.models.cnn_lstm import CNNLSTMModel
from pose_model.utils.logger import get_logger
from pose_model.utils.losses import FocalLoss
from pose_model.utils.metrics import confusion, per_class_accuracy, sequence_accuracy, sequence_f1
from pose_model.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CNN+LSTM head pose model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    accs, f1s = [], []
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
            total_loss += loss.item()
            accs.append(sequence_accuracy(logits, labels))
            f1s.append(sequence_f1(logits, labels))
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    mean_acc = sum(accs) / len(accs)
    mean_f1 = sum(f1s) / len(f1s)
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    cls_acc = per_class_accuracy(logits_cat, labels_cat, NUM_CLASSES)
    cm = confusion(logits_cat, labels_cat, NUM_CLASSES)
    return avg_loss, mean_acc, mean_f1, cls_acc, cm


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])

    checkpoint_path = args.checkpoint or cfg["eval"]["checkpoint"]
    if not checkpoint_path or not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = PoseSequenceDataset(
        data_root=cfg["data_root"],
        mode="test",
        sequence_length=cfg["sequence_length"],
        overlap=cfg["overlap"],
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        image_size=cfg["image_size"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = CNNLSTMModel(
        backbone=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    logger.info("Loaded checkpoint from %s", checkpoint_path)

    class_counts = [dataset.class_counts.get(i, 0) for i in range(NUM_CLASSES)]
    class_weights = compute_class_weights(class_counts, cfg["loss"], device)
    if cfg["loss"]["type"] == "focal":
        criterion = FocalLoss(gamma=cfg["loss"]["focal_gamma"], weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss, acc, f1, cls_acc, cm = evaluate(model, loader, criterion, device)
    logger.info("Test loss=%.4f acc=%.4f f1=%.4f", loss, acc, f1)
    logger.info("Per-class accuracy (0-6): %s", ["{:.3f}".format(x) for x in cls_acc])
    logger.info("Confusion matrix:\n%s", cm)


if __name__ == "__main__":
    main()
