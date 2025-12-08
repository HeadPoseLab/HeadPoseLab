import argparse
from pathlib import Path


MAP = {3: 2, 4: 3, 5: 4, 6: 5, 7: 5}


def remap_labels_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Remap labels in a single labels.txt file. Returns (total_lines, changed_lines)."""
    total = 0
    changed = 0
    new_lines = []

    with path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            try:
                filename, label_str = stripped.split(";", 1)
                label = int(label_str)
            except ValueError:
                new_lines.append(line)
                continue

            total += 1
            new_label = MAP.get(label, label)
            if new_label != label:
                changed += 1
            new_lines.append(f"{filename};{new_label}\n")

    if not dry_run:
        with path.open("w") as f:
            f.writelines(new_lines)

    return total, changed


def main():
    parser = argparse.ArgumentParser(
        description="Remap labels in pose_model/data/person_xxx/labels.txt (3->2, 4->3, 5->4, 6/7->5)."
    )
    parser.add_argument("--data-root", type=str, default="pose_model/data", help="Data root containing person_xxx dirs")
    parser.add_argument("--dry-run", action="store_true", help="Only report changes without writing files")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    person_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not person_dirs:
        raise RuntimeError(f"No person folders under {data_root}")

    total_files = 0
    total_labels = 0
    total_changed = 0
    for person in person_dirs:
        labels_path = person / "labels.txt"
        if not labels_path.exists():
            continue
        total_files += 1
        labels, changed = remap_labels_file(labels_path, dry_run=args.dry_run)
        total_labels += labels
        total_changed += changed
        status = "DRY-RUN" if args.dry_run else "UPDATED"
        print(f"[{status}] {labels_path}: {changed}/{labels} labels remapped")

    print(f"Processed {total_files} files, {total_changed}/{total_labels} labels remapped.")


if __name__ == "__main__":
    main()
