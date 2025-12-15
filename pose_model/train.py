import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pose_model.datasets.sequence_dataset import PoseSequenceDataset, NUM_CLASSES
from pose_model.models.cnn_lstm import CNNLSTMModel
from pose_model.utils.logger import get_logger
from pose_model.utils.losses import FocalLoss
from pose_model.utils.metrics import sequence_accuracy, sequence_f1
from pose_model.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM head pose model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
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


def build_dataset(cfg, mode: str, logger):
    try:
        dataset = PoseSequenceDataset(
            data_root=cfg["data_root"],
            mode=mode,
            sequence_length=cfg["sequence_length"],
            overlap=cfg["overlap"],
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            seed=cfg["seed"],
            image_size=cfg["image_size"],
        )
        return dataset
    except Exception as exc:  # noqa: BLE001
        if mode == "train":
            raise
        logger.warning("Skipping %s dataset: %s", mode, exc)
        return None


def build_dataloaders(cfg, logger):
    train_ds = build_dataset(cfg, "train", logger)
    val_ds = build_dataset(cfg, "val", logger)

    sampler = None
    if cfg.get("sampler", {}).get("balanced") and getattr(train_ds, "sample_weights", None):
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_ds.sample_weights, dtype=torch.double),
            num_samples=len(train_ds.sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
        )
        if val_ds
        else None
    )
    return train_loader, val_loader, train_ds, val_ds


def train_one_epoch(model, loader, optimizer, criterion, device, logger, grad_clip=None, log_interval: int = 10):
    model.train()
    total_loss = 0.0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)
        loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        if logger and step % max(1, log_interval) == 0:
            logger.info("step %d/%d | loss=%.4f", step, len(loader), loss.item())

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    accs, f1s = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))
            total_loss += loss.item()
            accs.append(sequence_accuracy(logits, labels))
            f1s.append(sequence_f1(logits, labels))
    return total_loss / len(loader), sum(accs) / len(accs), sum(f1s) / len(f1s)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    Path(cfg["train"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])

    logger.info("Using device: %s", device)
    train_loader, val_loader, train_ds, _ = build_dataloaders(cfg, logger)

    class_counts = [train_ds.class_counts.get(i, 0) for i in range(NUM_CLASSES)]
    class_weights = compute_class_weights(class_counts, cfg["loss"], device)

    model = CNNLSTMModel(
        backbone=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        bidirectional=cfg["model"].get("bidirectional", False),
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        freeze_stages=cfg["model"].get("freeze_stages", -1),
        pretrained=cfg["model"].get("pretrained", True),
        resnet_variant=cfg["model"].get("resnet_variant", "resnet18"),
        cnn_branch_channels=cfg["model"].get("cnn_branch_channels", None),
        fusion=cfg["model"].get("fusion", "concat"),
        fusion_dropout=cfg["model"].get("fusion_dropout", 0.0),
    ).to(device)

    writer = SummaryWriter(log_dir=cfg["train"].get("log_dir", "runs"))
    try:
        model.eval()
        example = torch.zeros(
            1, cfg["sequence_length"], 3, cfg["image_size"], cfg["image_size"], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            writer.add_graph(model, example)
        model.train()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping graph export to TensorBoard: %s", exc)


    if cfg["loss"]["type"] == "focal":
        criterion = FocalLoss(gamma=cfg["loss"]["focal_gamma"], weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    base_lr = cfg["train"]["lr"]
    backbone_lr_scale = cfg["train"].get("backbone_lr_scale", 0.1)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_scale})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=cfg["train"]["weight_decay"])

    best_val_loss = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            logger,
            grad_clip=cfg["train"]["grad_clip"],
            log_interval=cfg["train"]["log_interval"],
        )

        if val_loader:
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                epoch,
                train_loss,
                val_loss,
                val_acc,
                val_f1,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(cfg["train"]["save_dir"], "best.pt")
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt_path)
                logger.info("Saved best checkpoint to %s", ckpt_path)
        else:
            logger.info("Epoch %d | train_loss=%.4f", epoch, train_loss)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
