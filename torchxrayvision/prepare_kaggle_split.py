import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATA_ROOT = "/data/janthonio03/local_datasets/AIP/chest_xray"
DEFAULT_OUT_CSV = "/data/janthonio03/local_datasets/AIP/torchxrayvision/kaggle_split_seed42.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PERSON_PATTERN = re.compile(r"(person\d+)", re.IGNORECASE)

CLASSES = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a train/validation CSV split for Kaggle chest X-ray training data."
    )

    parser.add_argument(
        "--data_root",
        default=DEFAULT_DATA_ROOT,
        help="Path to Kaggle chest_xray root.",
    )

    parser.add_argument(
        "--out_csv",
        default=DEFAULT_OUT_CSV,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def extract_group(path):
    match = PERSON_PATTERN.search(path.name)
    return match.group(1).lower() if match else ""


def scan_training_images(data_root):
    records = []
    train_root = data_root / "train"

    for class_name, label in CLASSES.items():
        class_dir = train_root / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing expected directory: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            person_group = extract_group(image_path)
            split_group = person_group or f"file::{image_path.relative_to(data_root)}"

            records.append(
                {
                    "path": str(image_path),
                    "label": label,
                    "class_name": class_name,
                    "group": split_group,
                    "split_group": split_group,
                }
            )

    if not records:
        raise ValueError(f"No training images found under: {train_root}")

    return records


def assign_splits(records, val_ratio, seed):
    if not 0 < val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    grouped_by_label = defaultdict(dict)

    for record in records:
        label = record["label"]
        group_id = record["split_group"]
        grouped_by_label[label].setdefault(group_id, []).append(record)

    val_group_ids = set()

    for label, groups in grouped_by_label.items():
        group_items = list(groups.items())
        rng.shuffle(group_items)

        total_count = sum(len(group_records) for _, group_records in group_items)
        target_val_count = round(total_count * val_ratio)
        current_val_count = 0

        for group_id, group_records in group_items:
            if current_val_count >= target_val_count:
                break

            val_group_ids.add(group_id)
            current_val_count += len(group_records)

    for record in records:
        record["split"] = "val" if record["split_group"] in val_group_ids else "train"
        del record["split_group"]

    return records


def write_csv(records, out_csv):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["path", "label", "class_name", "group", "split"]

    with out_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records):
    counts = Counter((record["split"], record["class_name"]) for record in records)

    print("Split class counts:")

    for split in ("train", "val"):
        for class_name in CLASSES:
            print(f"{split},{class_name}: {counts[(split, class_name)]}")


def main():
    args = parse_args()

    data_root = Path(args.data_root).expanduser()
    out_csv = Path(args.out_csv).expanduser()

    records = scan_training_images(data_root)
    records = assign_splits(records, args.val_ratio, args.seed)

    write_csv(records, out_csv)
    print_summary(records)

    print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()