import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from models.torchxrayvision_model import DEFAULT_WEIGHTS, TorchXRayVisionDenseNetBinary
    from preprocessing import preprocess_for_model, read_image_as_rgb
except ImportError:
    from src.models.torchxrayvision_model import DEFAULT_WEIGHTS, TorchXRayVisionDenseNetBinary
    from src.preprocessing import preprocess_for_model, read_image_as_rgb


CASE_ORDER = ["TP", "FP", "FN", "TN"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM panels from saved prediction CSV files.")
    parser.add_argument("--checkpoint", required=True, help="TorchXRayVision checkpoint path.")
    parser.add_argument("--kaggle_predictions", required=True, help="Kaggle val prediction CSV.")
    parser.add_argument("--rsna_predictions", required=True, help="RSNA external prediction CSV.")
    parser.add_argument("--threshold_json", default=None, help="threshold_policies_seed*.json or internal_threshold_seed*.json.")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed Kaggle-derived threshold. Overrides threshold_json.")
    parser.add_argument("--out_dir", default="outputs/figures/gradcam")
    parser.add_argument("--kaggle_root", default=None, help="Fallback Kaggle chest_xray root for missing/relative prediction paths.")
    parser.add_argument("--rsna_root", default=None, help="Fallback RSNA root for missing/relative prediction paths.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples_per_case", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path_prefix_from", default=None, help="Optional prefix to replace in prediction CSV paths.")
    parser.add_argument("--path_prefix_to", default=None, help="Replacement prefix for prediction CSV paths.")
    parser.add_argument("--no_body_crop", action="store_true", help="Disable visualization-time body crop preprocessing.")
    return parser.parse_args()


def load_threshold(path):
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if item.get("name") == "youden_j":
                return float(item["threshold"])
    if isinstance(data, dict):
        if "threshold" in data:
            return float(data["threshold"])
        if "youden_j" in data and isinstance(data["youden_j"], dict):
            return float(data["youden_j"]["threshold"])
    raise ValueError(f"Could not find a threshold in {path}")


def get_device(device_arg):
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(checkpoint_path, weights, device):
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


def resolve_target_layer(model):
    features = model.backbone.features
    if hasattr(features, "denseblock4"):
        return features.denseblock4
    return features


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits.squeeze()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap = F.interpolate(heatmap, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()
        max_value = float(np.max(heatmap))
        if max_value > 0:
            heatmap = heatmap / max_value
        return heatmap

    def close(self):
        for handle in self.handles:
            handle.remove()


def read_predictions(path, threshold, prefix_from=None, prefix_to=None):
    df = pd.read_csv(path)
    if "label" not in df.columns and "true_label" in df.columns:
        df["label"] = df["true_label"]

    required = {"label", "prob", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    if threshold is not None:
        df["pred"] = (df["prob"].astype(float) >= threshold).astype(int)
    elif "pred" not in df.columns:
        raise ValueError(f"{path} has no pred column; provide --threshold or --threshold_json.")

    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["prob"] = df["prob"].astype(float)

    if prefix_from and prefix_to:
        df["path"] = df["path"].map(lambda p: str(p).replace(prefix_from, prefix_to, 1))
    return df


def _suffix_after_marker(path, markers):
    parts = Path(path).parts
    lowered = [part.lower() for part in parts]
    for marker in markers:
        marker_lower = marker.lower()
        matching = [idx for idx, part in enumerate(lowered) if part == marker_lower]
        if matching:
            idx = matching[-1]
            if idx + 1 < len(parts):
                return Path(*parts[idx + 1 :])
    return None


def resolve_image_path(path_value, root, domain_name, patient_id=None):
    path = Path(str(path_value))
    candidates = [path]

    if root:
        root = Path(root)
        if not path.is_absolute():
            candidates.append(root / path)

        if domain_name == "rsna_external":
            candidates.append(root / "stage_2_train_images" / path.name)
            if patient_id:
                candidates.append(root / "stage_2_train_images" / f"{patient_id}.dcm")
        else:
            suffix = _suffix_after_marker(path, ["chest_xray", "chest_xray_kaggle"])
            if suffix:
                candidates.append(root / suffix)
            for split in ("train", "val", "test"):
                if split in path.parts:
                    idx = path.parts.index(split)
                    candidates.append(root / Path(*path.parts[idx:]))
                    break

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return path


def add_case_column(df):
    conditions = [
        (df["label"] == 1) & (df["pred"] == 1),
        (df["label"] == 0) & (df["pred"] == 1),
        (df["label"] == 1) & (df["pred"] == 0),
        (df["label"] == 0) & (df["pred"] == 0),
    ]
    df = df.copy()
    df["case"] = np.select(conditions, CASE_ORDER, default="UNK")
    return df


def sample_cases(df, samples_per_case, seed):
    rng = random.Random(seed)
    sampled = []
    for case in CASE_ORDER:
        part = df[df["case"] == case].copy()
        if part.empty:
            continue
        if case in {"TP", "FP"}:
            part = part.sort_values("prob", ascending=False)
        else:
            part = part.sort_values("prob", ascending=True)
        top = part.head(max(samples_per_case * 5, samples_per_case))
        indices = list(top.index)
        rng.shuffle(indices)
        sampled.extend(indices[:samples_per_case])
    return df.loc[sampled].reset_index(drop=True)


def safe_stem(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value[:120].strip("_") or "sample"


def overlay_heatmap(image, heatmap, alpha=0.40):
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heatmap = np.asarray(heatmap, dtype=np.float32).clip(0.0, 1.0)
    color = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    color[..., 0] = np.clip(1.8 * heatmap, 0.0, 1.0)
    color[..., 1] = np.clip(1.8 * heatmap - 0.55, 0.0, 1.0)
    color[..., 2] = np.clip(0.65 - 1.2 * heatmap, 0.0, 1.0)
    overlay = (1.0 - alpha) * base + alpha * color
    overlay = (overlay.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(overlay)


def save_panel(original, processed, overlay, out_path):
    width, height = processed.size
    original_resized = original.resize((width, height), Image.BILINEAR)
    panel = Image.new("RGB", (width * 3, height), "white")
    panel.paste(original_resized, (0, 0))
    panel.paste(processed, (width, 0))
    panel.paste(overlay, (width * 2, 0))
    panel.save(out_path)


def generate_for_domain(domain_name, df, gradcam, device, args, domain_root):
    out_dir = Path(args.out_dir) / domain_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for i, row in df.iterrows():
        image_path = resolve_image_path(
            row["path"],
            domain_root,
            domain_name,
            patient_id=row.get("patient_id"),
        )
        if not image_path.is_file():
            print(f"[WARN] Missing image path, skipping: {image_path}")
            continue

        original = read_image_as_rgb(image_path)
        tensor, processed = preprocess_for_model(
            original,
            image_size=args.image_size,
            use_body_crop=not args.no_body_crop,
        )
        image_tensor = tensor.unsqueeze(0).to(device)
        heatmap = gradcam(image_tensor)
        overlay = overlay_heatmap(processed, heatmap)

        identifier = row.get("patient_id", image_path.stem)
        prefix = f"{i:03d}_{row['case']}_y{int(row['label'])}_p{int(row['pred'])}_prob{row['prob']:.4f}_{safe_stem(identifier)}"
        sample_dir = out_dir / prefix
        sample_dir.mkdir(parents=True, exist_ok=True)
        original.save(sample_dir / "original.png")
        processed.save(sample_dir / "preprocessed.png")
        overlay.save(sample_dir / "gradcam_overlay.png")
        save_panel(original, processed, overlay, sample_dir / "panel.png")

        manifest_rows.append(
            {
                "domain": domain_name,
                "case": row["case"],
                "label": int(row["label"]),
                "pred": int(row["pred"]),
                "prob": float(row["prob"]),
                "path": str(image_path),
                "output_dir": str(sample_dir),
            }
        )

    if manifest_rows:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)


def main():
    args = parse_args()
    threshold = args.threshold if args.threshold is not None else load_threshold(args.threshold_json)
    device = get_device(args.device)
    model = load_model(args.checkpoint, args.weights, device)
    gradcam = GradCAM(model, resolve_target_layer(model))

    try:
        kaggle_df = add_case_column(
            read_predictions(args.kaggle_predictions, threshold, args.path_prefix_from, args.path_prefix_to)
        )
        rsna_df = add_case_column(
            read_predictions(args.rsna_predictions, threshold, args.path_prefix_from, args.path_prefix_to)
        )
        generate_for_domain(
            "kaggle_internal",
            sample_cases(kaggle_df, args.samples_per_case, args.seed),
            gradcam,
            device,
            args,
            args.kaggle_root,
        )
        generate_for_domain(
            "rsna_external",
            sample_cases(rsna_df, args.samples_per_case, args.seed + 1),
            gradcam,
            device,
            args,
            args.rsna_root,
        )
    finally:
        gradcam.close()

    print(f"Saved Grad-CAM outputs to: {Path(args.out_dir)}")
    if threshold is not None:
        print(f"Applied fixed threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
