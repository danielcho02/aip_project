import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

try:
    from dataset import KaggleChestXrayDataset
    from models.torchxrayvision_model import (
        DEFAULT_WEIGHTS,
        TorchXRayVisionDenseNetBinary,
    )
    from rsna_dataset import RSNAPneumoniaDataset
except ImportError:
    from src.dataset import KaggleChestXrayDataset
    from src.models.torchxrayvision_model import (
        DEFAULT_WEIGHTS,
        TorchXRayVisionDenseNetBinary,
    )
    from src.rsna_dataset import RSNAPneumoniaDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TorchXRayVision DenseNet on Kaggle validation and RSNA external validation."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to trained TorchXRayVision checkpoint.")
    parser.add_argument("--kaggle_csv", required=True, help="Path to Kaggle split CSV.")
    parser.add_argument("--rsna_root", required=True, help="Path to RSNA root directory.")
    parser.add_argument("--out_dir", default="outputs/torchxrayvision_external", help="Output directory.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="TorchXRayVision pretrained weight name.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda or cpu.")
    parser.add_argument(
        "--sample_per_class",
        type=int,
        default=1000,
        help="RSNA samples per class. Use 100 or 200 for a quick test.",
    )
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


def predict(model, loader, device, include_patient_id=False):
    labels = []
    probs = []
    paths = []
    patient_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            batch_probs = torch.sigmoid(logits)

            labels.extend(batch["label"].cpu().numpy().tolist())
            probs.extend(batch_probs.cpu().numpy().tolist())
            paths.extend(batch["path"])
            if include_patient_id:
                patient_ids.extend(batch["patient_id"])

    result = {
        "labels": np.array(labels, dtype=np.int64),
        "probs": np.array(probs, dtype=np.float32),
        "paths": paths,
    }
    if include_patient_id:
        result["patient_ids"] = patient_ids
    return result


def calculate_youden_threshold(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_index = int(np.argmax(j_scores))
    return {
        "threshold": float(thresholds[best_index]),
        "youden_j": float(j_scores[best_index]),
        "tpr": float(tpr[best_index]),
        "fpr": float(fpr[best_index]),
    }


def calculate_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(np.int64)
    accuracy = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "threshold": float(threshold),
        "n_samples": int(len(labels)),
    }


def save_json(path, data):
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def save_rsna_predictions(path, predictions, threshold):
    preds = (predictions["probs"] >= threshold).astype(np.int64)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["patient_id", "label", "prob", "pred", "path"])
        writer.writeheader()
        for patient_id, label, prob, pred, image_path in zip(
            predictions["patient_ids"],
            predictions["labels"],
            predictions["probs"],
            preds,
            predictions["paths"],
        ):
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "label": int(label),
                    "prob": float(prob),
                    "pred": int(pred),
                    "path": image_path,
                }
            )


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    pin_memory = device.type == "cuda"
    model = load_torchxrayvision_model(args.checkpoint, args.weights, device)

    kaggle_dataset = KaggleChestXrayDataset(
        csv_path=args.kaggle_csv,
        split="val",
        image_size=args.image_size,
        augment=False,
    )
    kaggle_loader = DataLoader(
        kaggle_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Evaluating Kaggle val: n={len(kaggle_dataset)}")
    kaggle_predictions = predict(model, kaggle_loader, device)
    threshold_info = calculate_youden_threshold(
        kaggle_predictions["labels"],
        kaggle_predictions["probs"],
    )
    threshold = threshold_info["threshold"]
    internal_metrics = calculate_metrics(
        kaggle_predictions["labels"],
        kaggle_predictions["probs"],
        threshold,
    )

    rsna_dataset = RSNAPneumoniaDataset(
        rsna_root=args.rsna_root,
        sample_per_class=args.sample_per_class,
        image_size=args.image_size,
        seed=args.seed,
    )
    rsna_loader = DataLoader(
        rsna_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"Evaluating RSNA external: n={len(rsna_dataset)}, threshold={threshold:.6f}")
    rsna_predictions = predict(model, rsna_loader, device, include_patient_id=True)
    rsna_metrics = calculate_metrics(
        rsna_predictions["labels"],
        rsna_predictions["probs"],
        threshold,
    )

    threshold_path = out_dir / f"internal_threshold_seed{args.seed}.json"
    internal_metrics_path = out_dir / f"internal_metrics_seed{args.seed}.json"
    rsna_metrics_path = out_dir / f"rsna_external_metrics_seed{args.seed}.json"
    rsna_predictions_path = out_dir / f"rsna_predictions_seed{args.seed}.csv"

    save_json(threshold_path, threshold_info)
    save_json(internal_metrics_path, internal_metrics)
    save_json(rsna_metrics_path, rsna_metrics)
    save_rsna_predictions(rsna_predictions_path, rsna_predictions, threshold)

    print("Internal Kaggle val metrics:")
    print(json.dumps(internal_metrics, indent=2))
    print("RSNA external metrics:")
    print(json.dumps(rsna_metrics, indent=2))
    print(f"Saved threshold: {threshold_path}")
    print(f"Saved internal metrics: {internal_metrics_path}")
    print(f"Saved RSNA metrics: {rsna_metrics_path}")
    print(f"Saved RSNA predictions: {rsna_predictions_path}")


if __name__ == "__main__":
    main()
