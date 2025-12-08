import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


LABEL_NAMES = {
    1: "正",
    2: "下",
    3: "左",
    4: "右",
    5: "歪",
}


def parse_args():
    parser = argparse.ArgumentParser(description="统计数据集中姿态标签占比并输出柱状图")
    parser.add_argument("--data-root", type=str, default="pose_model/data", help="数据根目录（含 person_xxx）")
    parser.add_argument("--output", type=str, default="data_distribution.png", help="输出图片路径")
    return parser.parse_args()


def read_labels(person_dir: Path) -> Counter:
    label_file = person_dir / "labels.txt"
    if not label_file.exists():
        raise FileNotFoundError(f"labels.txt missing in {person_dir}")
    counter = Counter()
    with label_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, label_str = line.split(";")
                label_id = int(label_str)
                counter[label_id] += 1
            except ValueError as exc:  # noqa: BLE001
                raise ValueError(f"Invalid line in {label_file}: {line}") from exc
    return counter


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    total_counter = Counter()
    persons = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not persons:
        raise RuntimeError(f"No person folders under {data_root}")

    for person in persons:
        total_counter.update(read_labels(person))

    total_samples = sum(total_counter.values())
    if total_samples == 0:
        raise RuntimeError("No labels found.")

    labels = list(range(1, 6))
    counts = [total_counter.get(l, 0) for l in labels]
    percents = [c / total_samples * 100 for c in counts]
    names = [LABEL_NAMES.get(l, str(l)) for l in labels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, percents, color="#4C72B0")
    plt.ylabel("占比 (%)")
    plt.title("姿态标签占比分布")
    plt.ylim(0, max(percents) * 1.2 if percents else 1)
    for bar, pct in zip(bars, percents):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"已统计 {total_samples} 条标注，结果写入 {args.output}")
    for l, c, p in zip(labels, counts, percents):
        print(f"类 {l} ({LABEL_NAMES[l]}): {c} 张, {p:.2f}%")


if __name__ == "__main__":
    main()
