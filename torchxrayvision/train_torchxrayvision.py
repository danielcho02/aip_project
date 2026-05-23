import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from dataset import KaggleChestXrayDataset

try:
    from torchxrayvision_model import (
        DEFAULT_WEIGHTS,
        TorchXRayVisionDenseNetBinary,
    )
except ImportError:
    try:
        from models.torchxrayvision_model import (
            DEFAULT_WEIGHTS,
            TorchXRayVisionDenseNetBinary,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot find TorchXRayVisionDenseNetBinary. "
            "Put torchxrayvision_model.py in this folder, "
            "or put it in models/torchxrayvision_model.py."
        ) from exc


DEFAULT_BASE_DIR = Path("/data/janthonio03/local_datasets/AIP")
DEFAULT_CSV_PATH = DEFAULT_BASE_DIR / "torchxrayvision" / "kaggle_split_seed42.csv"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "torchxrayvision" / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a TorchXRayVision DenseNet on Kaggle chest X-ray split CSV."
    )

    parser.add_argument(
        "--csv_path",
        default=str(DEFAULT_CSV_PATH),
        help="Path to split CSV.",
    )

    parser.add_argument(
        "--out_dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="TorchXRayVision pretrained weight name.",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--disable_harmonize_preprocess", action="store_true")
    parser.add_argument("--disable_crop_body", action="store_true")
    parser.add_argument("--percentile_lower", type=float, default=1.0)
    parser.add_argument("--percentile_upper", type=float, default=99.0)
    parser.add_argument("--crop_threshold", type=int, default=5)
    parser.add_argument("--crop_margin", type=int, default=10)

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


def calculate_pos_weight(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df["split"] == "train"]

    if train_df.empty:
        raise ValueError("No train rows found in CSV.")

    positive_count = int((train_df["label"] == 1).sum())
    negative_count = int((train_df["label"] == 0).sum())

    if positive_count == 0:
        raise ValueError("Cannot compute pos_weight because train split has no positive samples.")

    return negative_count / positive_count


def make_dataset_kwargs(args):
    return {
        "harmonize_preprocess": not args.disable_harmonize_preprocess,
        "crop_body": not args.disable_crop_body,
        "percentile_lower": args.percentile_lower,
        "percentile_upper": args.percentile_upper,
        "crop_threshold": args.crop_threshold,
        "crop_margin": args.crop_margin,
    }


def make_loaders(args, pin_memory):
    dataset_kwargs = make_dataset_kwargs(args)

    train_dataset = KaggleChestXrayDataset(
        csv_path=args.csv_path,
        split="train",
        image_size=args.image_size,
        augment=True,
        **dataset_kwargs,
    )

    val_dataset = KaggleChestXrayDataset(
        csv_path=args.csv_path,
        split="val",
        image_size=args.image_size,
        augment=False,
        **dataset_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images).reshape(-1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    labels_np = np.array(all_labels, dtype=np.int64)
    probs_np = np.array(all_probs, dtype=np.float32).reshape(-1)
    preds_np = (probs_np >= 0.5).astype(np.int64)

    accuracy = float(accuracy_score(labels_np, preds_np))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np,
        preds_np,
        average="binary",
        zero_division=0,
    )

    try:
        auc = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / total_count,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
    }


def save_checkpoint(path, model, optimizer, epoch, val_metrics, args):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "args": vars(args),
    }

    torch.save(checkpoint, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    pin_memory = device.type == "cuda"

    train_loader, val_loader = make_loaders(args, pin_memory)

    pos_weight_value = calculate_pos_weight(args.csv_path)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = TorchXRayVisionDenseNetBinary(weights=args.weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -math.inf

    best_checkpoint_path = out_dir / f"best_torchxrayvision_seed{args.seed}.pt"
    metrics_path = out_dir / f"metrics_seed{args.seed}.json"

    history = []

    print(f"Device: {device}")
    print(f"Weights: {args.weights}")
    print(f"CSV path: {args.csv_path}")
    print(f"Output dir: {out_dir}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"pos_weight: {pos_weight_value:.6f}")
    print(f"harmonize_preprocess: {not args.disable_harmonize_preprocess}")
    print(f"crop_body: {not args.disable_crop_body}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }

        history.append(epoch_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if not math.isnan(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                args=args,
            )

        with metrics_path.open("w", encoding="utf-8") as json_file:
            json.dump(history, json_file, indent=2)

    if best_auc > -math.inf:
        print(f"Best val_auc: {best_auc:.4f}")
        print(f"Saved checkpoint: {best_checkpoint_path}")
    else:
        print("Best val_auc: nan")
        print("No checkpoint saved because val_auc could not be computed.")

    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()