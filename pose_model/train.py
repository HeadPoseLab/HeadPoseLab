import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
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
from pose_model.models.model_factory import build_pose_model, collect_backbone_modules, model_requires_images
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
        hand_roi_cfg = cfg.get("hand_roi", {})
        augment_cfg = cfg.get("augment", {})
        load_images = model_requires_images(cfg)
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
            hand_roi_enabled=hand_roi_cfg.get("enabled", False),
            hand_roi_expand=hand_roi_cfg.get("expand", 1.6),
            hand_roi_min_scale=hand_roi_cfg.get("min_scale", 0.2),
            augment_cfg=augment_cfg,
            load_images=load_images,
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


def _set_module_requires_grad(module, requires_grad: bool):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _apply_freeze_stages(module, freeze_stages: int):
    if freeze_stages is None or freeze_stages < 0:
        return
    if hasattr(module, "_freeze_stages"):
        module._freeze_stages(freeze_stages)
        return
    resnet_branch = getattr(module, "resnet_branch", None)
    if resnet_branch is not None and hasattr(resnet_branch, "_freeze_stages"):
        resnet_branch._freeze_stages(freeze_stages)


def _set_backbone_trainable(model, trainable: bool, freeze_stages: int):
    backbones = collect_backbone_modules(model)
    for backbone in backbones:
        _set_module_requires_grad(backbone, trainable)
        if trainable and freeze_stages is not None and freeze_stages >= 0:
            _apply_freeze_stages(backbone, freeze_stages)


def build_optimizer(model, cfg):
    base_lr = cfg["train"]["lr"]
    backbone_lr_scale = cfg["train"].get("backbone_lr_scale", 0.1)
    backbone_modules = set(collect_backbone_modules(model))
    backbone_params = [p for m in backbone_modules for p in m.parameters() if p.requires_grad]
    backbone_param_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_scale})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
    return optim.AdamW(param_groups, lr=base_lr, weight_decay=cfg["train"]["weight_decay"])


def build_scheduler(optimizer, cfg):
    scheduler_cfg = cfg["train"].get("scheduler", {})
    if not scheduler_cfg or not scheduler_cfg.get("enabled", False):
        return None
    scheduler_type = str(scheduler_cfg.get("type", "warmup_cosine")).lower()
    if scheduler_type != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    total_epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.05))
    warmup_epochs = max(0, min(warmup_epochs, max(total_epochs - 1, 0)))

    def _lr_lambda(epoch_idx: int):
        if total_epochs <= 1:
            return 1.0
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return max(1e-6, float(epoch_idx + 1) / float(warmup_epochs))
        progress = (epoch_idx - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


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
    for step, (head_images, hand_images, head_labels, hand_labels, head_coords, hand_coords) in enumerate(
        loader, start=1
    ):
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
    if val_loader is None:
        raise RuntimeError(
            "Validation split is empty (val_loader is None). "
            "Please increase val_ratio or dataset size before training."
        )

    head_counts = [train_ds.class_counts_head.get(i, 0) for i in range(HEAD_NUM_CLASSES)]
    hand_counts = [train_ds.class_counts_hand.get(i, 0) for i in range(HAND_NUM_CLASSES)]
    class_weights_head = compute_class_weights(head_counts, cfg["loss"], device)
    class_weights_hand = compute_class_weights(hand_counts, cfg["loss"], device)

    model, model_arch = build_pose_model(cfg, device=device)
    logger.info("Model architecture: %s", model_arch)

    writer = SummaryWriter(log_dir=cfg["train"].get("log_dir", "runs"))
    try:
        model.eval()
        example_head = torch.zeros(
            1, cfg["sequence_length"], 3, cfg["image_size"], cfg["image_size"], device=device, dtype=torch.float32
        )
        example_hand = torch.zeros(
            1, cfg["sequence_length"], 3, cfg["image_size"], cfg["image_size"], device=device, dtype=torch.float32
        )
        example_head_coords = torch.zeros(
            1, cfg["sequence_length"], 2, device=device, dtype=torch.float32
        )
        example_hand_coords = torch.zeros(
            1, cfg["sequence_length"], 4, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            writer.add_graph(model, (example_head, example_hand, example_head_coords, example_hand_coords))
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
    freeze_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))
    freeze_backbone = cfg["model"].get("freeze_backbone", False)
    if freeze_backbone and freeze_epochs > 0:
        logger.warning("freeze_backbone is true; freeze_backbone_epochs will be ignored.")
        freeze_epochs = 0

    if freeze_epochs > 0:
        _set_backbone_trainable(model, False, cfg["model"].get("freeze_stages", -1))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    if scheduler is not None:
        scheduler.step()

    selection_metric = str(cfg["train"].get("selection_metric", "val_loss")).lower()
    selection_mode = str(cfg["train"].get("selection_mode", "min")).lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError("train.selection_mode must be 'min' or 'max'")
    best_primary = float("-inf") if selection_mode == "max" else float("inf")
    best_val_loss = float("inf")
    early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
    epochs_without_improvement = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _set_backbone_trainable(model, True, cfg["model"].get("freeze_stages", -1))
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg)
            if scheduler is not None:
                scheduler.step()
            logger.info("Unfroze backbone at epoch %d and rebuilt optimizer.", epoch)
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
        val_loss, head_acc, hand_acc, head_f1, hand_f1 = evaluate(
            model,
            val_loader,
            criterion_head,
            criterion_hand,
            device,
            head_weight,
            hand_weight,
        )
        metric_values = {
            "val_loss": val_loss,
            "head_acc": head_acc,
            "hand_acc": hand_acc,
            "head_f1": head_f1,
            "hand_f1": hand_f1,
        }
        if selection_metric not in metric_values:
            raise ValueError(
                f"Unsupported selection metric: {selection_metric}. "
                f"Available: {sorted(metric_values.keys())}"
            )
        primary_value = metric_values[selection_metric]
        improved_primary = (
            primary_value > best_primary
            if selection_mode == "max"
            else primary_value < best_primary
        )

        if improved_primary:
            best_primary = primary_value
            epochs_without_improvement = 0
            ckpt_name = f"best_{selection_metric}.pt"
            primary_ckpt_path = os.path.join(cfg["train"]["save_dir"], ckpt_name)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metrics": metric_values,
                },
                primary_ckpt_path,
            )
            # Keep backward compatibility with previous default naming.
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metrics": metric_values,
                },
                os.path.join(cfg["train"]["save_dir"], "best.pt"),
            )
            logger.info(
                "Saved primary best checkpoint to %s (%s=%.4f)",
                primary_ckpt_path,
                selection_metric,
                primary_value,
            )
        else:
            epochs_without_improvement += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if selection_metric != "val_loss":
                best_val_ckpt = os.path.join(cfg["train"]["save_dir"], "best_val_loss.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "epoch": epoch,
                        "metrics": metric_values,
                    },
                    best_val_ckpt,
                )
                logger.info("Saved best val-loss checkpoint to %s", best_val_ckpt)

        last_ckpt = os.path.join(cfg["train"]["save_dir"], "last.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "metrics": metric_values,
            },
            last_ckpt,
        )

        if scheduler is not None:
            scheduler.step()
        lr_log = ", ".join(f"{g['lr']:.6g}" for g in optimizer.param_groups)
        logger.info(
            "Epoch %d | train_loss=%.4f val_loss=%.4f head_acc=%.4f hand_acc=%.4f head_f1=%.4f hand_f1=%.4f | lrs=[%s]",
            epoch,
            train_loss,
            val_loss,
            head_acc,
            hand_acc,
            head_f1,
            hand_f1,
            lr_log,
        )

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            logger.info(
                "Early stopping at epoch %d after %d epochs without improvement on %s.",
                epoch,
                epochs_without_improvement,
                selection_metric,
            )
            break

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
