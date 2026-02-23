import json
import math
import re
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

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
        augment_cfg: dict | None = None,
        load_images: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mode = mode
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.image_size = image_size
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        self.use_default_transform = transform is None
        self.transform = transform or transforms.Compose([self.resize, self.to_tensor])
        self.head_dir = head_dir
        self.hand_dir = hand_dir
        self.sample_weight_head = float(sample_weight_head)
        self.sample_weight_hand = float(sample_weight_hand)
        self.hand_roi_enabled = bool(hand_roi_enabled)
        self.hand_roi_expand = float(hand_roi_expand)
        self.hand_roi_min_scale = float(hand_roi_min_scale)
        self.augment_cfg = augment_cfg or {}
        self.load_images = bool(load_images)
        self.augment_enabled = bool(self.augment_cfg.get("enabled", False)) and self.mode == "train" and self.load_images

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

    @staticmethod
    def _swap_left_right_labels(labels: List[int]) -> List[int]:
        # zero-based class ids: 2=left, 3=right
        swapped = []
        for label in labels:
            if label == 2:
                swapped.append(3)
            elif label == 3:
                swapped.append(2)
            else:
                swapped.append(label)
        return swapped

    @staticmethod
    def _flip_head_coord(coord: List[float]) -> List[float]:
        if len(coord) < 2:
            return coord
        x, y = float(coord[0]), float(coord[1])
        if x <= 0 and y <= 0:
            return [0.0, 0.0]
        return [max(0.0, min(1.0, 1.0 - x)), y]

    @staticmethod
    def _flip_hand_coord(coord: List[float]) -> List[float]:
        if len(coord) < 4:
            return coord
        lx, ly, rx, ry = [float(v) for v in coord[:4]]

        def _flip_point(px: float, py: float) -> Tuple[float, float]:
            if px <= 0 and py <= 0:
                return 0.0, 0.0
            return max(0.0, min(1.0, 1.0 - px)), py

        nlx, nly = _flip_point(rx, ry)
        nrx, nry = _flip_point(lx, ly)
        return [nlx, nly, nrx, nry]

    @staticmethod
    def _apply_affine_to_point(
        x: float,
        y: float,
        angle_deg: float,
        translate_xy: Tuple[int, int],
        scale: float,
        shear_xy_deg: Tuple[float, float],
        width: int,
        height: int,
    ) -> Tuple[float, float]:
        # Approximate forward affine transform for keypoint coordinates.
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        tx, ty = float(translate_xy[0]), float(translate_xy[1])
        angle = math.radians(angle_deg)
        shx = math.radians(shear_xy_deg[0])
        shy = math.radians(shear_xy_deg[1])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        # Move to center
        px = x - cx
        py = y - cy
        # Scale
        px *= scale
        py *= scale
        # Shear x then y
        px, py = px + math.tan(shx) * py, py
        px, py = px, py + math.tan(shy) * px
        # Rotate
        px, py = cos_a * px - sin_a * py, sin_a * px + cos_a * py
        # Move back + translate
        px += cx + tx
        py += cy + ty
        return px, py

    def _apply_affine_to_head_coord(self, coord: List[float], affine_state: Dict) -> List[float]:
        if len(coord) < 2:
            return coord
        x, y = float(coord[0]), float(coord[1])
        if x <= 0 and y <= 0:
            return [0.0, 0.0]
        px = x * self.image_size
        py = y * self.image_size
        px, py = self._apply_affine_to_point(
            px,
            py,
            affine_state["angle"],
            affine_state["translate"],
            affine_state["scale"],
            affine_state["shear"],
            self.image_size,
            self.image_size,
        )
        return [
            max(0.0, min(1.0, px / self.image_size)),
            max(0.0, min(1.0, py / self.image_size)),
        ]

    def _apply_affine_to_hand_coord(self, coord: List[float], affine_state: Dict) -> List[float]:
        if len(coord) < 4:
            return coord
        out: List[float] = []
        for idx in (0, 2):
            x, y = float(coord[idx]), float(coord[idx + 1])
            if x <= 0 and y <= 0:
                out.extend([0.0, 0.0])
                continue
            px = x * self.image_size
            py = y * self.image_size
            px, py = self._apply_affine_to_point(
                px,
                py,
                affine_state["angle"],
                affine_state["translate"],
                affine_state["scale"],
                affine_state["shear"],
                self.image_size,
                self.image_size,
            )
            out.extend(
                [
                    max(0.0, min(1.0, px / self.image_size)),
                    max(0.0, min(1.0, py / self.image_size)),
                ]
            )
        return out

    def _sample_affine_state(self) -> Dict | None:
        affine_cfg = self.augment_cfg.get("affine", {})
        p = float(affine_cfg.get("p", 0.0))
        if p <= 0 or random.random() >= p:
            return None

        degrees = float(affine_cfg.get("degrees", 0.0))
        angle = random.uniform(-degrees, degrees)

        translate = affine_cfg.get("translate", 0.0)
        if isinstance(translate, (list, tuple)):
            tx_ratio = float(translate[0]) if len(translate) > 0 else 0.0
            ty_ratio = float(translate[1]) if len(translate) > 1 else tx_ratio
        else:
            tx_ratio = ty_ratio = float(translate)
        tx = int(round(random.uniform(-tx_ratio, tx_ratio) * self.image_size))
        ty = int(round(random.uniform(-ty_ratio, ty_ratio) * self.image_size))

        scale_min = float(affine_cfg.get("scale_min", 1.0))
        scale_max = float(affine_cfg.get("scale_max", 1.0))
        if scale_min > scale_max:
            scale_min, scale_max = scale_max, scale_min
        scale = random.uniform(scale_min, scale_max)

        shear = affine_cfg.get("shear", 0.0)
        if isinstance(shear, (list, tuple)):
            if len(shear) == 2:
                sx = random.uniform(float(shear[0]), float(shear[1]))
                sy = 0.0
            elif len(shear) >= 4:
                sx = random.uniform(float(shear[0]), float(shear[1]))
                sy = random.uniform(float(shear[2]), float(shear[3]))
            else:
                s = float(shear[0])
                sx = random.uniform(-s, s)
                sy = 0.0
        else:
            s = float(shear)
            sx = random.uniform(-s, s)
            sy = 0.0

        return {
            "angle": angle,
            "translate": (tx, ty),
            "scale": scale,
            "shear": (sx, sy),
        }

    def _sample_erasing_state(self) -> Dict | None:
        erase_cfg = self.augment_cfg.get("erasing", {})
        p = float(erase_cfg.get("p", 0.0))
        if p <= 0 or random.random() >= p:
            return None

        scale_min = float(erase_cfg.get("scale_min", 0.02))
        scale_max = float(erase_cfg.get("scale_max", 0.12))
        ratio_min = float(erase_cfg.get("ratio_min", 0.3))
        ratio_max = float(erase_cfg.get("ratio_max", 3.3))
        if scale_min > scale_max:
            scale_min, scale_max = scale_max, scale_min
        if ratio_min > ratio_max:
            ratio_min, ratio_max = ratio_max, ratio_min

        area = float(self.image_size * self.image_size)
        for _ in range(10):
            target_area = random.uniform(scale_min, scale_max) * area
            ratio = random.uniform(ratio_min, ratio_max)
            h = int(round(math.sqrt(target_area * ratio)))
            w = int(round(math.sqrt(target_area / ratio)))
            if h < self.image_size and w < self.image_size and h > 0 and w > 0:
                top = random.randint(0, self.image_size - h)
                left = random.randint(0, self.image_size - w)
                return {
                    "top": top,
                    "left": left,
                    "height": h,
                    "width": w,
                    "value": float(erase_cfg.get("value", 0.0)),
                }
        return None

    def _sample_augment_state(self) -> Dict:
        color_cfg = self.augment_cfg.get("color_jitter", {})
        b = float(color_cfg.get("brightness", 0.0))
        c = float(color_cfg.get("contrast", 0.0))
        s = float(color_cfg.get("saturation", 0.0))
        h = float(color_cfg.get("hue", 0.0))
        color_state = {
            "brightness": random.uniform(max(0.0, 1.0 - b), 1.0 + b) if b > 0 else None,
            "contrast": random.uniform(max(0.0, 1.0 - c), 1.0 + c) if c > 0 else None,
            "saturation": random.uniform(max(0.0, 1.0 - s), 1.0 + s) if s > 0 else None,
            "hue": random.uniform(-h, h) if h > 0 else None,
        }

        blur_cfg = self.augment_cfg.get("blur", {})
        blur_state = None
        blur_p = float(blur_cfg.get("p", 0.0))
        if blur_p > 0 and random.random() < blur_p:
            kernel_size = int(blur_cfg.get("kernel_size", 3))
            if kernel_size <= 1:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma_min = float(blur_cfg.get("sigma_min", 0.1))
            sigma_max = float(blur_cfg.get("sigma_max", 1.2))
            if sigma_min > sigma_max:
                sigma_min, sigma_max = sigma_max, sigma_min
            blur_state = {"kernel_size": kernel_size, "sigma": random.uniform(sigma_min, sigma_max)}

        return {
            "flip": random.random() < float(self.augment_cfg.get("hflip_p", 0.0)),
            "color": color_state,
            "affine": self._sample_affine_state(),
            "blur": blur_state,
            "erasing": self._sample_erasing_state(),
        }

    def _apply_seq_augment(self, image: Image.Image, aug_state: Dict) -> torch.Tensor:
        image = self.resize(image)

        if aug_state.get("flip"):
            image = TF.hflip(image)

        affine_state = aug_state.get("affine")
        if affine_state:
            image = TF.affine(
                image,
                angle=float(affine_state["angle"]),
                translate=tuple(affine_state["translate"]),
                scale=float(affine_state["scale"]),
                shear=tuple(affine_state["shear"]),
                interpolation=InterpolationMode.BILINEAR,
            )

        color_state = aug_state.get("color", {})
        if color_state.get("brightness") is not None:
            image = TF.adjust_brightness(image, float(color_state["brightness"]))
        if color_state.get("contrast") is not None:
            image = TF.adjust_contrast(image, float(color_state["contrast"]))
        if color_state.get("saturation") is not None:
            image = TF.adjust_saturation(image, float(color_state["saturation"]))
        if color_state.get("hue") is not None:
            image = TF.adjust_hue(image, float(color_state["hue"]))

        blur_state = aug_state.get("blur")
        if blur_state:
            image = TF.gaussian_blur(
                image,
                kernel_size=int(blur_state["kernel_size"]),
                sigma=float(blur_state["sigma"]),
            )

        tensor = self.to_tensor(image)
        erase_state = aug_state.get("erasing")
        if erase_state:
            top = int(erase_state["top"])
            left = int(erase_state["left"])
            height = int(erase_state["height"])
            width = int(erase_state["width"])
            value = float(erase_state["value"])
            tensor = tensor.clone()
            tensor[:, top : top + height, left : left + width] = value
        return tensor

    def _augment_head_coord(self, coord: List[float], aug_state: Dict) -> List[float]:
        out = coord
        if aug_state.get("flip"):
            out = self._flip_head_coord(out)
        affine_state = aug_state.get("affine")
        if affine_state:
            out = self._apply_affine_to_head_coord(out, affine_state)
        return out

    def _augment_hand_coord(self, coord: List[float], aug_state: Dict) -> List[float]:
        out = coord
        if aug_state.get("flip"):
            out = self._flip_hand_coord(out)
        affine_state = aug_state.get("affine")
        if affine_state:
            out = self._apply_affine_to_hand_coord(out, affine_state)
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        head_paths, hand_paths, head_labels, hand_labels, head_coords, hand_coords = self.samples[idx]
        head_labels_out = list(head_labels)
        hand_labels_out = list(hand_labels)
        head_coords_out = [list(coord) for coord in head_coords]
        hand_coords_out = [list(coord) for coord in hand_coords]

        if not self.load_images:
            head_images_tensor = torch.zeros(self.sequence_length, 3, 1, 1, dtype=torch.float32)
            hand_images_tensor = torch.zeros(self.sequence_length, 3, 1, 1, dtype=torch.float32)
        elif self.augment_enabled and self.use_default_transform:
            aug_state = self._sample_augment_state()
            head_images = [self._apply_seq_augment(Image.open(path).convert("RGB"), aug_state) for path in head_paths]
            hand_images = []
            for path, coords in zip(hand_paths, hand_coords_out):
                img = Image.open(path).convert("RGB")
                if self.hand_roi_enabled:
                    img = self._crop_hand_roi(img, coords)
                hand_images.append(self._apply_seq_augment(img, aug_state))
            if aug_state.get("flip"):
                head_labels_out = self._swap_left_right_labels(head_labels_out)
                hand_labels_out = self._swap_left_right_labels(hand_labels_out)
            head_coords_out = [self._augment_head_coord(coord, aug_state) for coord in head_coords_out]
            hand_coords_out = [self._augment_hand_coord(coord, aug_state) for coord in hand_coords_out]
            head_images_tensor = torch.stack(head_images, dim=0)
            hand_images_tensor = torch.stack(hand_images, dim=0)
        else:
            head_images = [self.transform(Image.open(path).convert("RGB")) for path in head_paths]
            hand_images = []
            for path, coords in zip(hand_paths, hand_coords_out):
                img = Image.open(path).convert("RGB")
                if self.hand_roi_enabled:
                    img = self._crop_hand_roi(img, coords)
                hand_images.append(self.transform(img))
            head_images_tensor = torch.stack(head_images, dim=0)
            hand_images_tensor = torch.stack(hand_images, dim=0)

        head_labels_tensor = torch.tensor(head_labels_out, dtype=torch.long)
        hand_labels_tensor = torch.tensor(hand_labels_out, dtype=torch.long)
        head_coords_tensor = torch.tensor(head_coords_out, dtype=torch.float32)
        hand_coords_tensor = torch.tensor(hand_coords_out, dtype=torch.float32)
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
