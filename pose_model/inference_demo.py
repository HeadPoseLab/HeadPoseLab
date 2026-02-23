import argparse
import json
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

from pose_model.datasets.multimodal_sequence_dataset import HEAD_NUM_CLASSES, HAND_NUM_CLASSES
from pose_model.models.model_factory import build_pose_model, model_requires_images

LABEL_MAP = {0: "正", 1: "下", 2: "左", 3: "右", 4: "歪"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference demo for head/hand pose model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--person_dir", type=str, required=True, help="Directory containing head_pose/hand_pose")
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


def _load_labels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"labels.json must be a list: {path}")
    return data


def _index_entries(entries):
    indexed = {}
    for entry in entries:
        image_name = entry.get("image")
        if not image_name:
            continue
        idx = _extract_index(image_name)
        if idx is None:
            continue
        indexed[idx] = entry
    return indexed


def _head_coords_from_entry(entry: dict):
    kp = entry.get("keypoints", {}).get("head", {})
    return [float(kp.get("x", 0.0)), float(kp.get("y", 0.0))]


def _hand_coords_from_entry(entry: dict):
    keypoints = entry.get("keypoints", {})
    left = keypoints.get("left_hand", {})
    right = keypoints.get("right_hand", {})
    return [
        float(left.get("x", 0.0)),
        float(left.get("y", 0.0)),
        float(right.get("x", 0.0)),
        float(right.get("y", 0.0)),
    ]


def prepare_aligned_sequences(head_dir: Path, hand_dir: Path, seq_len: int, image_size: int, load_images: bool):
    transform = None
    if load_images:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    head_labels_path = head_dir.parent / "labels.json"
    hand_labels_path = hand_dir.parent / "labels.json"
    if head_labels_path.exists() and hand_labels_path.exists():
        head_map = _index_entries(_load_labels(head_labels_path))
        hand_map = _index_entries(_load_labels(hand_labels_path))
        common_indices = sorted(set(head_map.keys()) & set(hand_map.keys()))
        if len(common_indices) < seq_len:
            raise ValueError(f"Not enough aligned frames. Need at least {seq_len}.")
        selected = common_indices[:seq_len]
        head_paths = [head_dir / head_map[i]["image"] for i in selected]
        hand_paths = [hand_dir / hand_map[i]["image"] for i in selected]
        head_coords = [_head_coords_from_entry(head_map[i]) for i in selected]
        hand_coords = [_hand_coords_from_entry(hand_map[i]) for i in selected]
    else:
        head_map = _collect_images(head_dir)
        hand_map = _collect_images(hand_dir)
        common_indices = sorted(set(head_map.keys()) & set(hand_map.keys()))
        if len(common_indices) < seq_len:
            raise ValueError(f"Not enough aligned frames. Need at least {seq_len}.")
        selected = common_indices[:seq_len]
        head_paths = [head_map[i] for i in selected]
        hand_paths = [hand_map[i] for i in selected]
        head_coords = [[0.0, 0.0] for _ in selected]
        hand_coords = [[0.0, 0.0, 0.0, 0.0] for _ in selected]

    head_images = [transform(Image.open(p).convert("RGB")) for p in head_paths] if load_images else None
    hand_images = [transform(Image.open(p).convert("RGB")) for p in hand_paths] if load_images else None
    return (
        torch.stack(head_images, dim=0) if head_images is not None else None,
        torch.stack(hand_images, dim=0) if hand_images is not None else None,
        torch.tensor(head_coords, dtype=torch.float32),
        torch.tensor(hand_coords, dtype=torch.float32),
        head_paths,
        hand_paths,
    )


def _majority_smooth(preds: list[int], window: int) -> list[int]:
    if window <= 1 or len(preds) <= 2:
        return preds
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    out: list[int] = []
    for i in range(len(preds)):
        l = max(0, i - half)
        r = min(len(preds), i + half + 1)
        span = preds[l:r]
        counts = {}
        for x in span:
            counts[x] = counts.get(x, 0) + 1
        # deterministic tie-break by class id
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out.append(best)
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_cfg = cfg.get("inference", {})
    hand_unknown_threshold = args.hand_unknown_threshold
    if hand_unknown_threshold is None:
        hand_unknown_threshold = float(inference_cfg.get("hand_unknown_threshold", 0.0))
    hand_unknown_class = args.hand_unknown_class
    if hand_unknown_class is None:
        hand_unknown_class = int(inference_cfg.get("hand_unknown_class", HAND_NUM_CLASSES))
    hand_unknown_index = max(0, min(hand_unknown_class - 1, HAND_NUM_CLASSES - 1))

    model, model_arch = build_pose_model(cfg, device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    person_dir = Path(args.person_dir)
    head_images_dir = person_dir / cfg.get("head_dir", "head_pose") / "images"
    hand_images_dir = person_dir / cfg.get("hand_dir", "hand_pose") / "images"
    load_images = model_requires_images(cfg)
    head_tensor, hand_tensor, head_coords, hand_coords, head_paths, hand_paths = prepare_aligned_sequences(
        head_images_dir, hand_images_dir, cfg["sequence_length"], cfg["image_size"], load_images=load_images
    )
    if head_tensor is not None and hand_tensor is not None:
        head_tensor = head_tensor.unsqueeze(0).to(device)
        hand_tensor = hand_tensor.unsqueeze(0).to(device)
    head_coords = head_coords.unsqueeze(0).to(device)
    hand_coords = hand_coords.unsqueeze(0).to(device)

    with torch.no_grad():
        head_logits, hand_logits = model(head_tensor, hand_tensor, head_coords, hand_coords)
        head_preds = head_logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        hand_probs = torch.softmax(hand_logits, dim=-1)
        hand_max_probs, hand_preds_tensor = hand_probs.max(dim=-1)
        if hand_unknown_threshold and hand_unknown_threshold > 0:
            hand_preds_tensor = hand_preds_tensor.clone()
            hand_preds_tensor[hand_max_probs < hand_unknown_threshold] = hand_unknown_index
        hand_preds = hand_preds_tensor.squeeze(0).cpu().tolist()

    smooth_cfg = inference_cfg.get("smoothing", {})
    if smooth_cfg.get("enabled", False):
        window = int(smooth_cfg.get("window", 5))
        head_preds = _majority_smooth(head_preds, window)
        hand_preds = _majority_smooth(hand_preds, window)

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
