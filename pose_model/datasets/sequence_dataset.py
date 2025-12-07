import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

NUM_CLASSES = 7


class PoseSequenceDataset(Dataset):
    """
    Dataset that reads person folders formatted as:
    person_xxx/
      images/
        head_00001.jpg
        ...
      labels.txt   (image_name;class_id with class_id in [1,7])
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
        filter_cfg: dict | None = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mode = mode
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.image_size = image_size
        self.filter_cfg = filter_cfg or {}
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        self.person_dirs = sorted([p for p in self.data_root.iterdir() if p.is_dir()])
        if not self.person_dirs:
            raise RuntimeError(f"No person folders found under {self.data_root}")

        self.samples: List[Tuple[List[Path], List[int]]] = []
        self.class_counts: Counter[int] = Counter()
        self.primary_counts: Counter[int] = Counter()
        self.primary_labels: List[int] = []
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
        primary_labels: List[int] = []
        for person_dir in target:
            entries = self._read_person_entries(person_dir)
            if len(entries) < self.sequence_length:
                continue
            for start in range(0, len(entries) - self.sequence_length + 1, stride):
                window = entries[start : start + self.sequence_length]
                paths, labels = zip(*window)
                labels_zero_based = [lbl - 1 for lbl in labels]
                primary_labels.append(self._primary_label(labels_zero_based))
                self.samples.append((list(paths), labels_zero_based))

        if self.filter_cfg.get("enabled"):
            factor = float(self.filter_cfg.get("max_factor", 3.0))
            rng = random.Random(self.filter_cfg.get("seed", seed))
            initial_counts = Counter(primary_labels)
            if initial_counts:
                min_count = min(initial_counts.values())
                max_per_class = max(int(min_count * factor), min_count)
                indices = list(range(len(self.samples)))
                rng.shuffle(indices)
                kept_samples: List[Tuple[List[Path], List[int]]] = []
                kept_primary: List[int] = []
                kept_counter: Counter[int] = Counter()
                for idx in indices:
                    lbl = primary_labels[idx]
                    if kept_counter[lbl] >= max_per_class:
                        continue
                    kept_counter[lbl] += 1
                    kept_samples.append(self.samples[idx])
                    kept_primary.append(lbl)
                self.samples = kept_samples
                primary_labels = kept_primary

        if not self.samples:
            raise RuntimeError(f"No sequences created for mode={self.mode}. Check data volume and sequence_length.")
        self.primary_labels = primary_labels
        self.primary_counts = Counter(primary_labels)

        frame_counter: Counter[int] = Counter()
        for _, labels in self.samples:
            frame_counter.update(labels)
        self.class_counts = frame_counter

        if self.primary_counts:
            for lbl in self.primary_labels:
                self.sample_weights.append(1.0 / max(1, self.primary_counts[lbl]))

    @staticmethod
    def _read_person_entries(person_dir: Path) -> List[Tuple[Path, int]]:
        label_file = person_dir / "labels.txt"
        if not label_file.exists():
            raise FileNotFoundError(f"labels.txt missing in {person_dir}")

        entries: List[Tuple[Path, int]] = []
        with label_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    filename, label_str = line.split(";")
                except ValueError as exc:
                    raise ValueError(f"Invalid label line in {label_file}: {line}") from exc
                label = int(label_str)
                img_path = person_dir / "images" / filename
                entries.append((img_path, label))
        return entries

    @staticmethod
    def _primary_label(labels: List[int]) -> int:
        """Choose a representative label for a sequence for filtering/sampling."""
        counter = Counter(labels)
        if counter:
            return counter.most_common(1)[0][0]
        return labels[len(labels) // 2] if labels else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, labels = self.samples[idx]
        images = [self.transform(Image.open(path).convert("RGB")) for path in paths]
        images_tensor = torch.stack(images, dim=0)  # [seq, C, H, W]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return images_tensor, labels_tensor
