import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Compute head/hand label distributions.")
    parser.add_argument("--data-root", type=str, default="pose_model/data", help="Data root containing person folders")
    parser.add_argument("--head-dir", type=str, default="head_pose", help="Head pose directory name")
    parser.add_argument("--hand-dir", type=str, default="hand_pose", help="Hand pose directory name")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes (labels are 1..N)")
    parser.add_argument(
        "--person-range",
        type=str,
        default=None,
        help="Person index range, e.g. 1-15 or 1,3,7-9 (based on digits in folder name)",
    )
    parser.add_argument("--output", type=str, default="data_distribution.png", help="Output plot path")
    return parser.parse_args()


def load_labels(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"labels.json must be a list: {path}")
    return data


def read_pose_labels(person_dir: Path, pose_dir: str) -> Counter:
    labels_path = person_dir / pose_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json missing in {labels_path.parent}")
    counter = Counter()
    entries = load_labels(labels_path)
    for entry in entries:
        label = entry.get("label")
        if isinstance(label, int):
            counter[label] += 1
    return counter


def summarize(counter: Counter, num_classes: int):
    labels = list(range(1, num_classes + 1))
    counts = [counter.get(l, 0) for l in labels]
    total = sum(counts)
    percents = [c / total * 100 if total else 0.0 for c in counts]
    return labels, counts, percents, total


def plot_distribution(ax, labels, percents, title: str):
    names = [str(l) for l in labels]
    bars = ax.bar(names, percents, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, max(percents) * 1.2 if percents else 1)
    for bar, pct in zip(bars, percents):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom")


def _extract_index(name: str) -> int | None:
    matches = re.findall(r"(\d+)", name)
    if not matches:
        return None
    return int(matches[-1])


def _parse_range(spec: str) -> set[int]:
    selected = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    persons = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not persons:
        raise RuntimeError(f"No person folders under {data_root}")

    if args.person_range:
        selected = _parse_range(args.person_range)
        filtered = []
        for person in persons:
            idx = _extract_index(person.name)
            if idx is not None and idx in selected:
                filtered.append(person)
        persons = filtered
        if not persons:
            raise RuntimeError(f"No person folders match range: {args.person_range}")

    head_counter = Counter()
    hand_counter = Counter()
    for person in persons:
        head_counter.update(read_pose_labels(person, args.head_dir))
        hand_counter.update(read_pose_labels(person, args.hand_dir))

    head_labels, head_counts, head_percents, head_total = summarize(head_counter, args.num_classes)
    hand_labels, hand_counts, hand_percents, hand_total = summarize(hand_counter, args.num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_distribution(axes[0], head_labels, head_percents, "Head pose distribution")
    plot_distribution(axes[1], hand_labels, hand_percents, "Hand pose distribution")
    fig.tight_layout()
    fig.savefig(args.output)

    print(f"Head labels: {head_total} samples")
    for l, c, p in zip(head_labels, head_counts, head_percents):
        print(f"  class {l}: {c} ({p:.2f}%)")
    print(f"Hand labels: {hand_total} samples")
    for l, c, p in zip(hand_labels, hand_counts, hand_percents):
        print(f"  class {l}: {c} ({p:.2f}%)")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
