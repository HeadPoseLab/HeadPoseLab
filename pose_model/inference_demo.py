import argparse
import sys
from pathlib import Path

import re

import torch
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pose_model.models.multi_task_model import MultiTaskPoseModel

LABEL_MAP = {0: "正", 1: "下", 2: "左", 3: "右", 4: "歪"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference demo for head/hand pose model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--person_dir", type=str, required=True, help="Directory containing head_pose/hand_pose")
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_index(filename: str) -> int | None:
    matches = re.findall(r"(\d+)", filename)
    if not matches:
        return None
    return int(matches[-1])


def _collect_images(images_dir: Path):
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    index_map = {}
    for path in image_paths:
        idx = _extract_index(path.name)
        if idx is None:
            continue
        index_map[idx] = path
    return index_map


def prepare_aligned_sequences(head_dir: Path, hand_dir: Path, seq_len: int, image_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    head_map = _collect_images(head_dir)
    hand_map = _collect_images(hand_dir)
    common_indices = sorted(set(head_map.keys()) & set(hand_map.keys()))
    if len(common_indices) < seq_len:
        raise ValueError(f"Not enough aligned frames. Need at least {seq_len}.")
    selected = common_indices[:seq_len]
    head_paths = [head_map[i] for i in selected]
    hand_paths = [hand_map[i] for i in selected]
    head_images = [transform(Image.open(p).convert("RGB")) for p in head_paths]
    hand_images = [transform(Image.open(p).convert("RGB")) for p in hand_paths]
    return torch.stack(head_images, dim=0), torch.stack(hand_images, dim=0), head_paths, hand_paths


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskPoseModel(
        backbone=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
        temporal_encoder=cfg["model"].get("temporal_encoder", "transformer"),
        tcn_channels=cfg["model"].get("tcn_channels", None),
        tcn_kernel=cfg["model"].get("tcn_kernel", 3),
        tcn_dilations=cfg["model"].get("tcn_dilations", None),
        tcn_dropout=cfg["model"].get("tcn_dropout", 0.2),
        transformer_cfg=cfg["model"].get("transformer", None),
        shared_backbone=cfg["model"].get("shared_backbone", False),
        shared_temporal=cfg["model"].get("shared_temporal", False),
        freeze_backbone=cfg["model"]["freeze_backbone"],
        freeze_stages=cfg["model"].get("freeze_stages", -1),
        pretrained=cfg["model"].get("pretrained", True),
        resnet_variant=cfg["model"].get("resnet_variant", "resnet50"),
        cnn_branch_channels=cfg["model"].get("cnn_branch_channels", None),
        fusion=cfg["model"].get("fusion", "concat"),
        fusion_dropout=cfg["model"].get("fusion_dropout", 0.0),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    person_dir = Path(args.person_dir)
    head_images_dir = person_dir / cfg.get("head_dir", "head_pose") / "images"
    hand_images_dir = person_dir / cfg.get("hand_dir", "hand_pose") / "images"
    head_tensor, hand_tensor, head_paths, hand_paths = prepare_aligned_sequences(
        head_images_dir, hand_images_dir, cfg["sequence_length"], cfg["image_size"]
    )
    head_tensor = head_tensor.unsqueeze(0).to(device)
    hand_tensor = hand_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        head_logits, hand_logits = model(head_tensor, hand_tensor)
        head_preds = head_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        hand_preds = hand_logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    print("Frame-level predictions:")
    for head_path, hand_path, head_pred, hand_pred in zip(head_paths, hand_paths, head_preds, hand_preds):
        head_label_id = head_pred + 1
        hand_label_id = hand_pred + 1
        head_name = LABEL_MAP.get(head_pred, "?")
        hand_name = LABEL_MAP.get(hand_pred, "?")
        print(
            f"{head_path.name} head={head_label_id} ({head_name}) | "
            f"{hand_path.name} hand={hand_label_id} ({hand_name})"
        )


if __name__ == "__main__":
    main()
