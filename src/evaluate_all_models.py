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
    from rsna_dataset import RSNAPneumoniaDataset
except ImportError:
    from src.dataset import KaggleChestXrayDataset
    from src.rsna_dataset import RSNAPneumoniaDataset


MODEL_CHOICES = ("baseline_cnn", "resnet50", "torchxrayvision")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation script for all models. "
            "Thresholds are selected only from Kaggle validation, then fixed for RSNA."
        )
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kaggle_csv", required=True)
    parser.add_argument("--rsna_root", required=True)
    parser.add_argument("--out_dir", default=None, help="Defaults to outputs/{model}_multi")
    parser.add_argument("--weights", default="densenet121-res224-chex", help="TorchXRayVision weights (ignored for other models)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_per_class", type=int, default=1000, help="RSNA samples per class. -1 for all.")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--recall_targets", type=float, nargs="+", default=[0.90, 0.95])
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def _load_state_dict(checkpoint_path, device):
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

    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def load_model(args, device):
    if args.model == "baseline_cnn":
        try:
            from models.baseline_cnn import BaselineCNN
        except ImportError:
            from src.models.baseline_cnn import BaselineCNN
        model = BaselineCNN()

    elif args.model == "resnet50":
        try:
            from models.resnet50 import ResNet50Binary
        except ImportError:
            from src.models.resnet50 import ResNet50Binary
        model = ResNet50Binary(pretrained=False)

    elif args.model == "torchxrayvision":
        try:
            from models.torchxrayvision_model import TorchXRayVisionDenseNetBinary
        except ImportError:
            from src.models.torchxrayvision_model import TorchXRayVisionDenseNetBinary
        model = TorchXRayVisionDenseNetBinary(weights=args.weights)

    state_dict = _load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model, loader, device, include_patient_id=False):
    labels, probs, paths, patient_ids = [], [], [], []

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


# ---------------------------------------------------------------------------
# Threshold policy builders
# ---------------------------------------------------------------------------

def _build_youden_policy(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return {"name": "youden_j", "threshold": float(thresholds[idx]), "youden_j": float(j[idx])}


def _build_f1_max_policy(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best_f1, best_thr = -1.0, 0.5
    for thr in thresholds:
        preds = (probs >= thr).astype(np.int64)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return {"name": "f1_max", "threshold": best_thr, "f1_on_kaggle_val": best_f1}


def _build_recall_target_policy(labels, probs, target):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    candidates = [(thr, r) for thr, r in zip(thresholds, tpr) if r >= target]
    if candidates:
        thr, achieved_recall = max(candidates, key=lambda x: x[0])
    else:
        thr, achieved_recall = float(thresholds[-1]), float(tpr[-1])
    name = f"recall_target_{target:.2f}".rstrip("0").rstrip(".")
    return {"name": name, "threshold": float(thr), "achieved_recall_on_kaggle_val": float(achieved_recall)}


def build_threshold_policies(labels, probs, recall_targets):
    policies = [{"name": "default_0.5", "threshold": 0.5}]
    policies.append(_build_youden_policy(labels, probs))
    policies.append(_build_f1_max_policy(labels, probs))
    for t in recall_targets:
        policies.append(_build_recall_target_policy(labels, probs, t))
    seen = {}
    for p in policies:
        seen[p["name"]] = p
    return list(seen.values())


# ---------------------------------------------------------------------------
# Metrics + Bootstrap CI
# ---------------------------------------------------------------------------

def _flatten_probs(probs):
    arr = np.asarray(probs, dtype=np.float64)
    return arr.ravel()


def calculate_metrics(labels, probs, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probs = _flatten_probs(probs)
    preds = (probs >= threshold).astype(np.int64)

    accuracy = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
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
        "true_positive_rate": float(labels.mean()),
        "predicted_positive_rate": float(preds.mean()),
        "mean_probability": float(probs.mean()),
    }


def bootstrap_ci(labels, probs, threshold, n_bootstrap=1000, seed=42):
    labels = np.asarray(labels, dtype=np.int64)
    probs = _flatten_probs(probs)

    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(labels)
    buckets = {k: [] for k in ("accuracy", "precision", "recall", "f1", "auc", "predicted_positive_rate", "mean_probability")}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sl, sp = labels[idx], probs[idx]
        if len(np.unique(sl)) < 2:
            continue
        m = calculate_metrics(sl, sp, threshold)
        for k in buckets:
            buckets[k].append(m[k])

    ci = {}
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=np.float64)
        if len(arr) == 0:
            ci[k] = {"mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
        else:
            ci[k] = {
                "mean": float(np.mean(arr)),
                "lower_95": float(np.percentile(arr, 2.5)),
                "upper_95": float(np.percentile(arr, 97.5)),
            }
    ci["n_valid_bootstrap"] = int(len(buckets["auc"]))
    return ci


def evaluate_with_policies(labels, probs, policies, n_bootstrap, seed):
    report = {}
    for policy in policies:
        name = policy["name"]
        thr = policy["threshold"]
        metrics = calculate_metrics(labels, probs, thr)
        ci = bootstrap_ci(labels, probs, thr, n_bootstrap=n_bootstrap, seed=seed)
        report[name] = {"policy": policy, "metrics": metrics, "bootstrap_95_ci": ci}
    return report


def make_domain_shift_summary(internal_report, external_report):
    summary = {}
    for name in internal_report:
        im = internal_report[name]["metrics"]
        em = external_report[name]["metrics"]
        summary[name] = {
            "internal_auc": im["auc"],
            "external_auc": em["auc"],
            "auc_drop": im["auc"] - em["auc"],
            "internal_f1": im["f1"],
            "external_f1": em["f1"],
            "f1_drop": im["f1"] - em["f1"],
            "internal_recall": im["recall"],
            "external_recall": em["recall"],
            "recall_drop": im["recall"] - em["recall"],
            "internal_precision": im["precision"],
            "external_precision": em["precision"],
            "precision_drop": im["precision"] - em["precision"],
            "internal_predicted_positive_rate": im["predicted_positive_rate"],
            "external_predicted_positive_rate": em["predicted_positive_rate"],
        }
    return summary


def print_policy_table(title, report):
    print(f"\n[{title}]")
    header = f"{'policy':<28} {'thr':>8} {'auc':>8} {'f1':>8} {'recall':>8} {'prec':>8} {'acc':>8}"
    print(header)
    print("-" * len(header))
    for name, entry in report.items():
        m = entry["metrics"]
        print(
            f"{name:<28} {m['threshold']:>8.4f} {m['auc']:>8.4f} {m['f1']:>8.4f} "
            f"{m['recall']:>8.4f} {m['precision']:>8.4f} {m['accuracy']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_predictions(path, predictions, threshold, include_patient_id=False):
    preds = (predictions["probs"] >= threshold).astype(np.int64)
    fieldnames = ["label", "prob", "pred", "path"]
    if include_patient_id:
        fieldnames = ["patient_id"] + fieldnames

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(predictions["labels"])):
            row = {
                "label": int(predictions["labels"][i]),
                "prob": float(predictions["probs"][i]),
                "pred": int(preds[i]),
                "path": predictions["paths"][i],
            }
            if include_patient_id:
                row["patient_id"] = predictions["patient_ids"][i]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/{args.model}_multi")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    print(f"[INFO] Model: {args.model}, Checkpoint: {args.checkpoint}")
    model = load_model(args, device)

    kaggle_dataset = KaggleChestXrayDataset(
        csv_path=args.kaggle_csv, split="val", image_size=args.image_size, augment=False
    )
    kaggle_loader = DataLoader(
        kaggle_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    rsna_dataset = RSNAPneumoniaDataset(
        rsna_root=args.rsna_root, image_size=args.image_size,
        sample_per_class=args.sample_per_class if args.sample_per_class > 0 else None,
        seed=args.seed,
    )
    rsna_loader = DataLoader(
        rsna_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    print(f"[INFO] Kaggle val: n={len(kaggle_dataset)}")
    kaggle_preds = predict(model, kaggle_loader, device)

    print(f"[INFO] RSNA external: n={len(rsna_dataset)}")
    rsna_preds = predict(model, rsna_loader, device, include_patient_id=True)

    policies = build_threshold_policies(kaggle_preds["labels"], kaggle_preds["probs"], args.recall_targets)

    print(f"\n[INFO] Threshold policies built from Kaggle val (n_bootstrap={args.n_bootstrap})...")
    internal_report = evaluate_with_policies(
        kaggle_preds["labels"], kaggle_preds["probs"], policies, args.n_bootstrap, args.seed
    )
    external_report = evaluate_with_policies(
        rsna_preds["labels"], rsna_preds["probs"], policies, args.n_bootstrap, args.seed
    )
    domain_shift_summary = make_domain_shift_summary(internal_report, external_report)

    main_policy_name = "youden_j"
    main_threshold = internal_report[main_policy_name]["metrics"]["threshold"]

    save_json(out_dir / f"threshold_policies_seed{args.seed}.json", policies)
    save_json(out_dir / f"internal_report_seed{args.seed}.json", internal_report)
    save_json(out_dir / f"rsna_external_report_seed{args.seed}.json", external_report)
    save_json(out_dir / f"domain_shift_summary_seed{args.seed}.json", domain_shift_summary)
    save_predictions(out_dir / f"kaggle_val_predictions_seed{args.seed}.csv", kaggle_preds, main_threshold)
    save_predictions(out_dir / f"rsna_predictions_seed{args.seed}.csv", rsna_preds, main_threshold, include_patient_id=True)

    print_policy_table("Internal Kaggle Validation", internal_report)
    print_policy_table("External RSNA Validation", external_report)

    print("\n[Domain Shift Summary]")
    print(json.dumps(domain_shift_summary, indent=2))

    print(f"\n[INFO] Main policy: {main_policy_name}, threshold: {main_threshold:.6f}")
    print(f"[INFO] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
