import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from models.torchxrayvision_model import DEFAULT_WEIGHTS, TorchXRayVisionDenseNetBinary
except ImportError:
    from src.models.torchxrayvision_model import DEFAULT_WEIGHTS, TorchXRayVisionDenseNetBinary


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_data_root():
    user = os.environ.get("USER", "")
    return f"/local_datasets/{user}/chest_xray_kaggle/chest_xray"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Kaggle validation predictions from an existing TorchXRayVision checkpoint."
    )
    parser.add_argument("--checkpoint", default="outputs/torchxrayvision/best_torchxrayvision_seed42.pt")
    parser.add_argument("--split_csv", default="outputs/splits/kaggle_split_seed42.csv")
    parser.add_argument("--threshold_json", default="outputs/torchxrayvision_external/internal_threshold_seed42.json")
    parser.add_argument("--data_root", default=default_data_root())
    parser.add_argument("--out_dir", default="outputs/torchxrayvision_external")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_torchxrayvision_model(checkpoint_path, weights, device):
    model = TorchXRayVisionDenseNetBinary(weights=weights)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_threshold(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "threshold" in data:
        return float(data["threshold"]), data
    if isinstance(data, list):
        for policy in data:
            if policy.get("name") == "youden_j" and "threshold" in policy:
                return float(policy["threshold"]), policy
    raise ValueError(f"Could not find threshold in {path}")


def _suffix_after_marker(path, markers):
    parts = Path(path).parts
    lowered = [part.lower() for part in parts]
    for marker in markers:
        matching = [idx for idx, part in enumerate(lowered) if part == marker.lower()]
        if matching:
            idx = matching[-1]
            if idx + 1 < len(parts):
                return Path(*parts[idx + 1 :])
    return None


def resolve_kaggle_path(path_value, data_root):
    path = Path(str(path_value))
    candidates = [path]
    data_root = Path(data_root)

    if not path.is_absolute():
        candidates.append(data_root / path)

    suffix = _suffix_after_marker(path, ["chest_xray", "chest_xray_kaggle"])
    if suffix:
        candidates.append(data_root / suffix)

    for split_name in ("train", "val", "test"):
        if split_name in path.parts:
            idx = path.parts.index(split_name)
            candidates.append(data_root / Path(*path.parts[idx:]))
            break

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


class KaggleValPredictionDataset(Dataset):
    def __init__(self, split_csv, data_root, image_size=224):
        self.split_csv = Path(split_csv)
        self.data_root = Path(data_root)
        df = pd.read_csv(self.split_csv)

        required = {"path", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.split_csv}: {', '.join(sorted(missing))}")

        self.df = df[df["split"] == "val"].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No validation rows found in {self.split_csv}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = resolve_kaggle_path(row["path"], self.data_root)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Kaggle validation image: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        image_id = str(row.get("group", "")) or image_path.stem
        return {
            "image": image,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "id": image_id,
            "path": str(image_path),
            "class_name": str(row.get("class_name", "")),
        }


def predict(model, loader, device):
    ids = []
    labels = []
    probs = []
    paths = []
    class_names = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            batch_probs = torch.sigmoid(logits)

            ids.extend(batch["id"])
            labels.extend(batch["label"].cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())
            paths.extend(batch["path"])
            class_names.extend(batch["class_name"])

    return {
        "ids": ids,
        "labels": np.asarray(labels, dtype=np.int64),
        "probs": np.asarray(probs, dtype=np.float32),
        "paths": paths,
        "class_names": class_names,
    }


def save_predictions(path, predictions, threshold):
    preds = (predictions["probs"] >= threshold).astype(np.int64)
    fieldnames = ["id", "true_label", "label", "prob", "pred", "path", "class_name"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(predictions["labels"])):
            label = int(predictions["labels"][i])
            writer.writerow(
                {
                    "id": predictions["ids"][i],
                    "true_label": label,
                    "label": label,
                    "prob": float(predictions["probs"][i]),
                    "pred": int(preds[i]),
                    "path": predictions["paths"][i],
                    "class_name": predictions["class_names"][i],
                }
            )


def save_threshold_policy(path, threshold, threshold_info):
    policy = {"name": "youden_j", "threshold": float(threshold)}
    if isinstance(threshold_info, dict):
        for key in ("youden_j", "tpr", "fpr"):
            if key in threshold_info:
                policy[key] = float(threshold_info[key])

    with path.open("w", encoding="utf-8") as f:
        json.dump([policy], f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold, threshold_info = load_threshold(args.threshold_json)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    model = load_torchxrayvision_model(args.checkpoint, args.weights, device)
    dataset = KaggleValPredictionDataset(args.split_csv, args.data_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Kaggle validation samples: {len(dataset)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold source: {args.threshold_json}")
    print(f"Fixed threshold: {threshold:.6f}")

    predictions = predict(model, loader, device)
    predictions_path = out_dir / f"kaggle_val_predictions_seed{args.seed}.csv"
    policies_path = out_dir / f"threshold_policies_seed{args.seed}.json"

    save_predictions(predictions_path, predictions, threshold)
    save_threshold_policy(policies_path, threshold, threshold_info)

    print(f"Saved Kaggle predictions: {predictions_path}")
    print(f"Saved threshold policies: {policies_path}")
    print("RSNA prediction CSV is not modified; existing file is reused for Grad-CAM.")


if __name__ == "__main__":
    main()
