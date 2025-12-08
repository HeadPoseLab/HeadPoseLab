import argparse
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pose_model.models.cnn_lstm import CNNLSTMModel

LABEL_MAP = {0: "正", 1: "下", 2: "左", 3: "右", 4: "歪"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference demo for head pose model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing a sequence of images (will be sorted by name)",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_sequence(images_dir: Path, seq_len: int, image_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if len(image_paths) < seq_len:
        raise ValueError(f"Not enough images in {images_dir}. Need at least {seq_len}.")
    image_paths = image_paths[:seq_len]
    images = [transform(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(images, dim=0), image_paths


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTMModel(
        backbone=cfg["model"]["backbone"],
        feature_dim=cfg["model"]["feature_dim"],
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    seq_tensor, image_paths = prepare_sequence(Path(args.images_dir), cfg["sequence_length"], cfg["image_size"])
    seq_tensor = seq_tensor.unsqueeze(0).to(device)  # [1, seq, C, H, W]

    with torch.no_grad():
        logits, _ = model(seq_tensor)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    print("Frame-level predictions:")
    for path, pred in zip(image_paths, preds):
        label_id = pred + 1
        label_name = LABEL_MAP.get(pred, "?")
        print(f"{path.name}: class={label_id} ({label_name})")


if __name__ == "__main__":
    main()
