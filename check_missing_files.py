import argparse
import json
from pathlib import Path


def load_labels(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"labels.json must be a list: {path}")
    return data


def collect_missing(person_dir: Path, pose_dir: str):
    missing_labels = []
    missing_image_dirs = []
    missing_images = set()
    invalid_labels = []

    base_dir = person_dir / pose_dir
    labels_path = base_dir / "labels.json"
    images_dir = base_dir / "images"

    if not labels_path.exists():
        missing_labels.append(str(labels_path))
        return missing_labels, missing_image_dirs, missing_images, invalid_labels

    if not images_dir.exists():
        missing_image_dirs.append(str(images_dir))
        return missing_labels, missing_image_dirs, missing_images, invalid_labels

    try:
        entries = load_labels(labels_path)
    except Exception as exc:  # noqa: BLE001
        invalid_labels.append(f"{labels_path} ({exc})")
        return missing_labels, missing_image_dirs, missing_images, invalid_labels

    for entry in entries:
        image_name = entry.get("image")
        if not image_name:
            continue
        image_path = images_dir / image_name
        if not image_path.exists():
            missing_images.add(str(image_path))

    return missing_labels, missing_image_dirs, missing_images, invalid_labels


def main():
    parser = argparse.ArgumentParser(description="Check missing images referenced by labels.json.")
    parser.add_argument("--data-root", type=str, default="pose_model/data", help="Data root containing person folders")
    parser.add_argument("--head-dir", type=str, default="head_pose", help="Head pose directory name")
    parser.add_argument("--hand-dir", type=str, default="hand_pose", help="Hand pose directory name")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    persons = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not persons:
        raise RuntimeError(f"No person folders under {data_root}")

    missing_labels = []
    missing_image_dirs = []
    missing_images = set()
    invalid_labels = []

    for person in persons:
        for pose_dir in (args.head_dir, args.hand_dir):
            m_labels, m_dirs, m_imgs, invalid = collect_missing(person, pose_dir)
            missing_labels.extend(m_labels)
            missing_image_dirs.extend(m_dirs)
            missing_images.update(m_imgs)
            invalid_labels.extend(invalid)

    print(f"Scanned {len(persons)} person folders under {data_root}")
    print(f"Missing labels.json files: {len(missing_labels)}")
    print(f"Missing images/ directories: {len(missing_image_dirs)}")
    print(f"Invalid labels.json files: {len(invalid_labels)}")
    print(f"Missing images referenced by labels.json: {len(missing_images)}")

    if missing_labels:
        print("\nMissing labels.json:")
        for path in sorted(missing_labels):
            print(path)

    if missing_image_dirs:
        print("\nMissing images/ directories:")
        for path in sorted(missing_image_dirs):
            print(path)

    if invalid_labels:
        print("\nInvalid labels.json:")
        for entry in sorted(invalid_labels):
            print(entry)

    if missing_images:
        print("\nMissing images:")
        for path in sorted(missing_images):
            print(path)


if __name__ == "__main__":
    main()
