import json
import re
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

HEAD_NUM_CLASSES = 5
HAND_NUM_CLASSES = 4

_INDEX_RE = re.compile(r"(\d+)")


def _extract_index(filename: str) -> int | None:
    matches = _INDEX_RE.findall(filename)
    if not matches:
        return None
    return int(matches[-1])


class MultiPoseSequenceDataset(Dataset):
    """
    Dataset for paired head/hand pose sequences:
    person_xx/
      head_pose/images/*.jpg + head_pose/labels.json
      hand_pose/images/*.jpg + hand_pose/labels.json
    labels.json item: {"image": "...", "label": int, "keypoints": {...}}
    """

    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        sequence_length: int = 16,
        overlap: bool = False,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        image_size: int = 128,
        transform=None,
        head_dir: str = "head_pose",
        hand_dir: str = "hand_pose",
        sample_weight_head: float = 0.5,
        sample_weight_hand: float = 0.5,
        hand_roi_enabled: bool = False,
        hand_roi_expand: float = 1.6,
        hand_roi_min_scale: float = 0.2,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mode = mode
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.image_size = image_size
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.head_dir = head_dir
        self.hand_dir = hand_dir
        self.sample_weight_head = float(sample_weight_head)
        self.sample_weight_hand = float(sample_weight_hand)
        self.hand_roi_enabled = bool(hand_roi_enabled)
        self.hand_roi_expand = float(hand_roi_expand)
        self.hand_roi_min_scale = float(hand_roi_min_scale)

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        self.person_dirs = sorted([p for p in self.data_root.iterdir() if p.is_dir()])
        if not self.person_dirs:
            raise RuntimeError(f"No person folders found under {self.data_root}")

        self.samples: List[
            Tuple[List[Path], List[Path], List[int], List[int], List[List[float]], List[List[float]]]
        ] = []
        self.class_counts_head: Counter[int] = Counter()
        self.class_counts_hand: Counter[int] = Counter()
        self.sample_weights: List[float] = []
        self._build_index(train_ratio, val_ratio, seed)

    def _build_index(self, train_ratio: float, val_ratio: float, seed: int):
        rng = random.Random(seed)
        persons = self.person_dirs.copy()
        rng.shuffle(persons)

        total = len(persons)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        if self.mode == "train":
            target = persons[:n_train]
        elif self.mode == "val":
            target = persons[n_train : n_train + n_val]
        elif self.mode == "test":
            target = persons[n_train + n_val :]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        stride = 1 if self.overlap else self.sequence_length
        head_counter: Counter[int] = Counter()
        hand_counter: Counter[int] = Counter()
        for person_dir in target:
            aligned = self._read_aligned_entries(person_dir)
            if len(aligned) < self.sequence_length:
                continue
            for start in range(0, len(aligned) - self.sequence_length + 1, stride):
                window = aligned[start : start + self.sequence_length]
                head_paths, hand_paths, head_labels, hand_labels, head_coords, hand_coords = zip(*window)
                head_labels_zero = [lbl - 1 for lbl in head_labels]
                hand_labels_zero = [lbl - 1 for lbl in hand_labels]
                head_counter.update(head_labels_zero)
                hand_counter.update(hand_labels_zero)
                self.samples.append(
                    (
                        list(head_paths),
                        list(hand_paths),
                        list(head_labels_zero),
                        list(hand_labels_zero),
                        list(head_coords),
                        list(hand_coords),
                    )
                )

        if not self.samples:
            raise RuntimeError(f"No sequences created for mode={self.mode}. Check data volume and sequence_length.")
        self.class_counts_head = head_counter
        self.class_counts_hand = hand_counter
        if self.class_counts_head and self.class_counts_hand:
            weight_total = self.sample_weight_head + self.sample_weight_hand
            if weight_total <= 0:
                weight_total = 1.0
            for _, _, head_labels, hand_labels, _, _ in self.samples:
                head_weights = [1.0 / max(1, self.class_counts_head[label]) for label in head_labels]
                hand_weights = [1.0 / max(1, self.class_counts_hand[label]) for label in hand_labels]
                avg_head = sum(head_weights) / len(head_weights)
                avg_hand = sum(hand_weights) / len(hand_weights)
                weighted = (
                    self.sample_weight_head * avg_head + self.sample_weight_hand * avg_hand
                ) / weight_total
                self.sample_weights.append(weighted)

    def _read_aligned_entries(
        self, person_dir: Path
    ) -> List[Tuple[Path, Path, int, int, List[float], List[float]]]:
        head_root = person_dir / self.head_dir
        hand_root = person_dir / self.hand_dir
        head_labels_path = head_root / "labels.json"
        hand_labels_path = hand_root / "labels.json"
        if not head_labels_path.exists() or not hand_labels_path.exists():
            raise FileNotFoundError(f"labels.json missing in {person_dir}")

        head_entries = self._load_labels(head_labels_path)
        hand_entries = self._load_labels(hand_labels_path)
        head_map = self._index_entries(head_entries)
        hand_map = self._index_entries(hand_entries)

        common_indices = sorted(set(head_map.keys()) & set(hand_map.keys()))
        aligned: List[Tuple[Path, Path, int, int, List[float], List[float]]] = []
        for idx in common_indices:
            head_item = head_map[idx]
            hand_item = hand_map[idx]
            head_label = self._validate_head_label(head_item.get("label"))
            hand_label = self._remap_hand_label(hand_item.get("label"))
            head_image = head_root / "images" / head_item["image"]
            hand_image = hand_root / "images" / hand_item["image"]
            aligned.append(
                (
                    head_image,
                    hand_image,
                    head_label,
                    hand_label,
                    self._flatten_head_coords(head_item.get("keypoints", {})),
                    self._flatten_hand_coords(hand_item.get("keypoints", {})),
                )
            )
        return aligned

    @staticmethod
    def _load_labels(path: Path) -> List[Dict]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"labels.json must be a list: {path}")
        return data

    @staticmethod
    def _index_entries(entries: List[Dict]) -> Dict[int, Dict]:
        indexed: Dict[int, Dict] = {}
        for entry in entries:
            image_name = entry.get("image")
            if not image_name:
                continue
            idx = _extract_index(image_name)
            if idx is None:
                continue
            indexed[idx] = entry
        return indexed

    @staticmethod
    def _flatten_head_coords(keypoints: Dict) -> List[float]:
        head = keypoints.get("head", {})
        return [float(head.get("x", 0.0)), float(head.get("y", 0.0))]

    @staticmethod
    def _flatten_hand_coords(keypoints: Dict) -> List[float]:
        left = keypoints.get("left_hand", {})
        right = keypoints.get("right_hand", {})
        return [
            float(left.get("x", 0.0)),
            float(left.get("y", 0.0)),
            float(right.get("x", 0.0)),
            float(right.get("y", 0.0)),
        ]

    @staticmethod
    def _validate_head_label(label) -> int:
        if not isinstance(label, int) or not (1 <= label <= HEAD_NUM_CLASSES):
            raise ValueError(f"Invalid head label: {label}")
        return label

    @staticmethod
    def _remap_hand_label(label) -> int:
        if not isinstance(label, int):
            raise ValueError(f"Invalid hand label: {label}")
        if label == 2:
            raise ValueError("Hand label 2 is not supported (class removed).")
        if label < 1 or label > 5:
            raise ValueError(f"Invalid hand label: {label}")
        return label if label < 2 else label - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        head_paths, hand_paths, head_labels, hand_labels, head_coords, hand_coords = self.samples[idx]
        head_images = [self.transform(Image.open(path).convert("RGB")) for path in head_paths]
        hand_images = []
        for path, coords in zip(hand_paths, hand_coords):
            img = Image.open(path).convert("RGB")
            if self.hand_roi_enabled:
                img = self._crop_hand_roi(img, coords)
            hand_images.append(self.transform(img))
        head_images_tensor = torch.stack(head_images, dim=0)
        hand_images_tensor = torch.stack(hand_images, dim=0)
        head_labels_tensor = torch.tensor(head_labels, dtype=torch.long)
        hand_labels_tensor = torch.tensor(hand_labels, dtype=torch.long)
        head_coords_tensor = torch.tensor(head_coords, dtype=torch.float32)
        hand_coords_tensor = torch.tensor(hand_coords, dtype=torch.float32)
        return (
            head_images_tensor,
            hand_images_tensor,
            head_labels_tensor,
            hand_labels_tensor,
            head_coords_tensor,
            hand_coords_tensor,
        )

    def _crop_hand_roi(self, image: Image.Image, coords: List[float]) -> Image.Image:
        if len(coords) < 4:
            return image
        lx, ly, rx, ry = coords[:4]
        points = []
        if lx > 0 or ly > 0:
            points.append((lx, ly))
        if rx > 0 or ry > 0:
            points.append((rx, ry))
        if not points:
            return image
        width, height = image.size
        xs = [p[0] * width for p in points]
        ys = [p[1] * height for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        box_w = max_x - min_x
        box_h = max_y - min_y
        min_side = min(width, height)
        min_size = self.hand_roi_min_scale * min_side
        size = max(box_w, box_h, min_size) * self.hand_roi_expand
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        half = size / 2.0
        left = max(0.0, cx - half)
        top = max(0.0, cy - half)
        right = min(width, cx + half)
        bottom = min(height, cy + half)
        if right <= left or bottom <= top:
            return image
        return image.crop((left, top, right, bottom))
