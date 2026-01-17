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

from pose_model.datasets.multimodal_sequence_dataset import (
    MultiPoseSequenceDataset,
    HEAD_NUM_CLASSES,
    HAND_NUM_CLASSES,
)
from pose_model.models.multi_task_model import MultiTaskPoseModel
from pose_model.utils.logger import get_logger
from pose_model.utils.losses import FocalLoss
from pose_model.utils.metrics import sequence_accuracy, sequence_f1
from pose_model.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train head/hand pose model")
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
        sampler_cfg = cfg.get("sampler", {})
        dataset = MultiPoseSequenceDataset(
            data_root=cfg["data_root"],
            mode=mode,
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion_head,
    criterion_hand,
    device,
    logger,
    grad_clip=None,
    log_interval: int = 10,
    head_weight: float = 1.0,
    hand_weight: float = 1.0,
):
    model.train()
    total_loss = 0.0
    for step, (head_images, hand_images, head_labels, hand_labels, _, _) in enumerate(loader, start=1):
        head_images = head_images.to(device)
        hand_images = hand_images.to(device)
        head_labels = head_labels.to(device)
        hand_labels = hand_labels.to(device)

        head_logits, hand_logits = model(head_images, hand_images)
        head_loss = criterion_head(head_logits.view(-1, HEAD_NUM_CLASSES), head_labels.view(-1))
        hand_loss = criterion_hand(hand_logits.view(-1, HAND_NUM_CLASSES), hand_labels.view(-1))
        loss = head_weight * head_loss + hand_weight * hand_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        if logger and step % max(1, log_interval) == 0:
            logger.info(
                "step %d/%d | loss=%.4f head=%.4f hand=%.4f",
                step,
                len(loader),
                loss.item(),
                head_loss.item(),
                hand_loss.item(),
            )

    return total_loss / len(loader)


def evaluate(model, loader, criterion_head, criterion_hand, device, head_weight: float, hand_weight: float):
    model.eval()
    total_loss = 0.0
    head_accs, hand_accs = [], []
    head_f1s, hand_f1s = [], []
    with torch.no_grad():
        for head_images, hand_images, head_labels, hand_labels, _, _ in loader:
            head_images = head_images.to(device)
            hand_images = hand_images.to(device)
            head_labels = head_labels.to(device)
            hand_labels = hand_labels.to(device)
            head_logits, hand_logits = model(head_images, hand_images)
            head_loss = criterion_head(head_logits.view(-1, HEAD_NUM_CLASSES), head_labels.view(-1))
            hand_loss = criterion_hand(hand_logits.view(-1, HAND_NUM_CLASSES), hand_labels.view(-1))
            loss = head_weight * head_loss + hand_weight * hand_loss
            total_loss += loss.item()
            head_accs.append(sequence_accuracy(head_logits, head_labels))
            hand_accs.append(sequence_accuracy(hand_logits, hand_labels))
            head_f1s.append(sequence_f1(head_logits, head_labels))
            hand_f1s.append(sequence_f1(hand_logits, hand_labels))
    return (
        total_loss / len(loader),
        sum(head_accs) / len(head_accs),
        sum(hand_accs) / len(hand_accs),
        sum(head_f1s) / len(head_f1s),
        sum(hand_f1s) / len(hand_f1s),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    Path(cfg["train"]["save_dir"]).mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])

    logger.info("Using device: %s", device)
    train_loader, val_loader, train_ds, _ = build_dataloaders(cfg, logger)

    head_counts = [train_ds.class_counts_head.get(i, 0) for i in range(HEAD_NUM_CLASSES)]
    hand_counts = [train_ds.class_counts_hand.get(i, 0) for i in range(HAND_NUM_CLASSES)]
    class_weights_head = compute_class_weights(head_counts, cfg["loss"], device)
    class_weights_hand = compute_class_weights(hand_counts, cfg["loss"], device)

    model = MultiTaskPoseModel(
        backbone=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
        temporal_encoder=cfg["model"].get("temporal_encoder", "transformer"),
        head_temporal_encoder=cfg["model"].get("head_temporal_encoder", None),
        hand_temporal_encoder=cfg["model"].get("hand_temporal_encoder", None),
        tcn_channels=cfg["model"].get("tcn_channels", None),
        tcn_kernel=cfg["model"].get("tcn_kernel", 3),
        tcn_dilations=cfg["model"].get("tcn_dilations", None),
        tcn_dropout=cfg["model"].get("tcn_dropout", 0.2),
        head_tcn_channels=cfg["model"].get("head_tcn_channels", None),
        hand_tcn_channels=cfg["model"].get("hand_tcn_channels", None),
        head_tcn_kernel=cfg["model"].get("head_tcn_kernel", None),
        hand_tcn_kernel=cfg["model"].get("hand_tcn_kernel", None),
        head_tcn_dilations=cfg["model"].get("head_tcn_dilations", None),
        hand_tcn_dilations=cfg["model"].get("hand_tcn_dilations", None),
        head_tcn_dropout=cfg["model"].get("head_tcn_dropout", None),
        hand_tcn_dropout=cfg["model"].get("hand_tcn_dropout", None),
        transformer_cfg=cfg["model"].get("transformer", None),
        head_transformer_cfg=cfg["model"].get("head_transformer", None),
        hand_transformer_cfg=cfg["model"].get("hand_transformer", None),
        shared_backbone=cfg["model"].get("shared_backbone", False),
        shared_temporal=cfg["model"].get("shared_temporal", False),
        adapter_enabled=cfg["model"].get("adapter", {}).get("enabled", False),
        adapter_dim=cfg["model"].get("adapter", {}).get("dim", None),
        adapter_dropout=cfg["model"].get("adapter", {}).get("dropout", 0.1),
        num_head_classes=cfg["model"].get("num_head_classes", HEAD_NUM_CLASSES),
        num_hand_classes=cfg["model"].get("num_hand_classes", HAND_NUM_CLASSES),
        freeze_backbone=cfg["model"]["freeze_backbone"],
        freeze_stages=cfg["model"].get("freeze_stages", -1),
        pretrained=cfg["model"].get("pretrained", True),
        resnet_variant=cfg["model"].get("resnet_variant", "resnet50"),
        cnn_branch_channels=cfg["model"].get("cnn_branch_channels", None),
        fusion=cfg["model"].get("fusion", "concat"),
        fusion_dropout=cfg["model"].get("fusion_dropout", 0.0),
    ).to(device)

    writer = SummaryWriter(log_dir=cfg["train"].get("log_dir", "runs"))
    try:
        model.eval()
        example_head = torch.zeros(
            1, cfg["sequence_length"], 3, cfg["image_size"], cfg["image_size"], device=device, dtype=torch.float32
        )
        example_hand = torch.zeros(
            1, cfg["sequence_length"], 3, cfg["image_size"], cfg["image_size"], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            writer.add_graph(model, (example_head, example_hand))
        model.train()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping graph export to TensorBoard: %s", exc)

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
    base_lr = cfg["train"]["lr"]
    backbone_lr_scale = cfg["train"].get("backbone_lr_scale", 0.1)
    backbone_modules = {model.backbone_head, model.backbone_hand}
    backbone_params = [p for m in backbone_modules for p in m.parameters() if p.requires_grad]
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
            criterion_head,
            criterion_hand,
            device,
            logger,
            grad_clip=cfg["train"]["grad_clip"],
            log_interval=cfg["train"]["log_interval"],
            head_weight=head_weight,
            hand_weight=hand_weight,
        )

        if val_loader:
            val_loss, head_acc, hand_acc, head_f1, hand_f1 = evaluate(
                model,
                val_loader,
                criterion_head,
                criterion_hand,
                device,
                head_weight,
                hand_weight,
            )
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f head_acc=%.4f hand_acc=%.4f head_f1=%.4f hand_f1=%.4f",
                epoch,
                train_loss,
                val_loss,
                head_acc,
                hand_acc,
                head_f1,
                hand_f1,
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
