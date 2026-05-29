import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required for src/visualize_results.py. Install it in the active env.") from exc


MODEL_ORDER = ["baseline_cnn", "resnet50", "torchxrayvision"]
MODEL_LABELS = {
    "baseline_cnn": "Baseline CNN",
    "baseline": "Baseline CNN",
    "resnet50": "ResNet50",
    "torchxrayvision": "TorchXRayVision",
}
MAIN_POLICY = "youden_j"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved AIP evaluation outputs without retraining.")
    parser.add_argument("--outputs_root", default="outputs", help="Primary outputs directory.")
    parser.add_argument("--external_outputs_root", default="external_outputs", help="Optional external outputs directory.")
    parser.add_argument("--out_dir", default="outputs/figures", help="Directory for generated figures.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--main_policy", default=MAIN_POLICY)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_entry(report, policy):
    if not report:
        return None
    if policy in report:
        return report[policy]
    if len(report) == 1:
        return next(iter(report.values()))
    return None


def ci_for(entry, metric):
    ci = (entry or {}).get("bootstrap_95_ci", {}).get(metric)
    if not ci:
        return None
    lower = ci.get("lower_95")
    upper = ci.get("upper_95")
    if lower is None or upper is None or math.isnan(float(lower)) or math.isnan(float(upper)):
        return None
    value = (entry or {}).get("metrics", {}).get(metric)
    if value is None:
        return None
    return max(0.0, value - lower), max(0.0, upper - value)


def infer_model_name(path):
    name = path.name.lower()
    if "baseline" in name:
        return "baseline_cnn"
    if "resnet50" in name:
        return "resnet50"
    if "torchxrayvision" in name:
        return "torchxrayvision"
    return name


def infer_variant(path):
    name = path.name.lower()
    if "focal" in name and ("crop" in name or "preprocess" in name):
        return "Preprocess + body crop + focal"
    if "focal" in name:
        return "Focal loss"
    if "crop" in name or "preprocess" in name:
        return "Preprocess + body crop"
    if "torchxrayvision" in name:
        return "TorchXRayVision"
    return MODEL_LABELS.get(infer_model_name(path), path.name)


def find_result_dirs(roots, seed):
    dirs = []
    required_names = {
        f"internal_report_seed{seed}.json",
        f"rsna_external_report_seed{seed}.json",
        f"domain_shift_summary_seed{seed}.json",
        f"threshold_policies_seed{seed}.json",
        f"internal_metrics_seed{seed}.json",
        f"rsna_external_metrics_seed{seed}.json",
    }
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.rglob("*"):
            if child.is_dir() and any((child / name).is_file() for name in required_names):
                dirs.append(child)
    return sorted(set(dirs))


def read_result_dir(path, seed, main_policy):
    internal_report_path = path / f"internal_report_seed{seed}.json"
    external_report_path = path / f"rsna_external_report_seed{seed}.json"
    threshold_path = path / f"threshold_policies_seed{seed}.json"

    model = infer_model_name(path)
    variant = infer_variant(path)
    row = {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "variant": variant,
        "dir": str(path),
        "has_ci": False,
        "has_predictions": (path / f"kaggle_val_predictions_seed{seed}.csv").is_file()
        and (path / f"rsna_predictions_seed{seed}.csv").is_file(),
        "has_thresholds": threshold_path.is_file(),
    }

    if internal_report_path.is_file() and external_report_path.is_file():
        internal_report = load_json(internal_report_path)
        external_report = load_json(external_report_path)
        internal_entry = metric_entry(internal_report, main_policy)
        external_entry = metric_entry(external_report, main_policy)
        if not internal_entry or not external_entry:
            return None, []

        for prefix, entry in [("internal", internal_entry), ("external", external_entry)]:
            for metric, value in entry["metrics"].items():
                row[f"{prefix}_{metric}"] = value
            for metric in ["auc", "f1", "recall", "precision"]:
                ci = ci_for(entry, metric)
                if ci:
                    row[f"{prefix}_{metric}_err_low"] = ci[0]
                    row[f"{prefix}_{metric}_err_high"] = ci[1]
                    row["has_ci"] = True

        policy_rows = []
        for policy, entry in external_report.items():
            metrics = entry["metrics"]
            policy_rows.append(
                {
                    "model": model,
                    "model_label": row["model_label"],
                    "variant": variant,
                    "policy": policy,
                    "threshold": metrics.get("threshold"),
                    "external_f1": metrics.get("f1"),
                    "external_recall": metrics.get("recall"),
                    "external_precision": metrics.get("precision"),
                    "external_auc": metrics.get("auc"),
                }
            )
        row["auc_drop"] = row.get("internal_auc", np.nan) - row.get("external_auc", np.nan)
        return row, policy_rows

    internal_metrics_path = path / f"internal_metrics_seed{seed}.json"
    external_metrics_path = path / f"rsna_external_metrics_seed{seed}.json"
    if internal_metrics_path.is_file() and external_metrics_path.is_file():
        internal = load_json(internal_metrics_path)
        external = load_json(external_metrics_path)
        for metric, value in internal.items():
            row[f"internal_{metric}"] = value
        for metric, value in external.items():
            row[f"external_{metric}"] = value
        row["auc_drop"] = row.get("internal_auc", np.nan) - row.get("external_auc", np.nan)
        return row, []

    return None, []


def ordered_main_results(df):
    rows = []
    for model in MODEL_ORDER:
        candidates = df[(df["model"] == model) & (df["variant"] == MODEL_LABELS.get(model, model))]
        if candidates.empty:
            candidates = df[df["model"] == model]
        if not candidates.empty:
            rows.append(candidates.iloc[0])
    return pd.DataFrame(rows)


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def err_array(df, prefix, metric):
    low = df.get(f"{prefix}_{metric}_err_low")
    high = df.get(f"{prefix}_{metric}_err_high")
    if low is None or high is None:
        return None
    values = np.vstack([low.fillna(0).to_numpy(), high.fillna(0).to_numpy()])
    return values if np.any(values > 0) else None


def save_internal_external_auc(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(df))
    width = 0.36
    ax.bar(
        x - width / 2,
        df["internal_auc"],
        width,
        yerr=err_array(df, "internal", "auc"),
        capsize=3,
        label="Kaggle internal",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        df["external_auc"],
        width,
        yerr=err_array(df, "external", "auc"),
        capsize=3,
        label="RSNA external",
        color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_label"], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("Internal vs external AUC (95% CI where available)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "internal_vs_external_auc.png")
    plt.close(fig)


def save_external_metric_comparison(df, out_dir):
    metrics = [("external_f1", "F1"), ("external_recall", "Recall"), ("external_precision", "Precision")]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(df))
    width = 0.24
    colors = ["#4C78A8", "#54A24B", "#E45756"]
    for i, (col, label) in enumerate(metrics):
        metric = col.replace("external_", "")
        yerr = err_array(df, "external", metric)
        offset = (i - 1) * width
        ax.bar(x + offset, df[col], width, yerr=yerr, capsize=3, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_label"], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("External F1 / Recall / Precision (95% CI where available)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "external_f1_recall_precision_by_model.png")
    plt.close(fig)


def save_auc_drop(df, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.bar(df["model_label"], df["auc_drop"], color="#B279A2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Internal AUC - External AUC")
    ax.set_title("AUC drop from Kaggle internal to RSNA external")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "auc_drop_by_model.png")
    plt.close(fig)


def save_threshold_policy_plot(policy_df, out_dir):
    if policy_df.empty:
        return
    policy_order = ["default_0.5", "youden_j", "f1_max", "recall_target_0.9", "recall_target_0.95"]
    policy_df = policy_df[policy_df["variant"].isin(policy_df["model_label"])].copy()
    if policy_df.empty:
        return
    policy_df["policy"] = pd.Categorical(policy_df["policy"], categories=policy_order, ordered=True)
    policy_df = policy_df.sort_values(["model", "policy"])

    metrics = [("external_f1", "External F1"), ("external_recall", "External Recall"), ("external_precision", "External Precision")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    colors = {"baseline_cnn": "#4C78A8", "resnet50": "#F58518", "torchxrayvision": "#54A24B"}

    for ax, (metric, title) in zip(axes, metrics):
        for model in MODEL_ORDER:
            part = policy_df[policy_df["model"] == model]
            if part.empty:
                continue
            ax.plot(part["policy"].astype(str), part[metric], marker="o", label=MODEL_LABELS[model], color=colors[model])
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Score")
    axes[-1].legend(frameon=False)
    fig.suptitle("External metrics by threshold policy")
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_policy_external_metrics.png")
    plt.close(fig)


def save_ablation_plot(df, out_dir):
    ablation = df[df["model"] == "torchxrayvision"].copy()
    if ablation.empty:
        return
    ablation = ablation.drop_duplicates(subset=["variant"], keep="first")
    order = ["TorchXRayVision", "Preprocess + body crop", "Focal loss", "Preprocess + body crop + focal"]
    ablation["variant"] = pd.Categorical(ablation["variant"], categories=order, ordered=True)
    ablation = ablation.sort_values("variant")

    metrics = [("external_auc", "AUC"), ("external_f1", "F1"), ("external_recall", "Recall"), ("external_precision", "Precision")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(ablation))
    width = 0.18
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for i, (col, label) in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, ablation[col], width, label=label, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(ablation["variant"].astype(str), rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("External score")
    ax.set_title("TorchXRayVision ablation on RSNA external (single-seed variants shown without CI)")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "preprocess_crop_focal_ablation.png")
    plt.close(fig)


def write_manifest(df, policy_df, out_dir):
    df.to_csv(out_dir / "visualized_result_dirs.csv", index=False)
    if not policy_df.empty:
        policy_df.to_csv(out_dir / "threshold_policy_metrics.csv", index=False)


def main():
    args = parse_args()
    roots = [Path(args.outputs_root), Path(args.external_outputs_root)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    rows = []
    policy_rows = []
    for result_dir in find_result_dirs(roots, args.seed):
        row, policies = read_result_dir(result_dir, args.seed, args.main_policy)
        if row:
            rows.append(row)
            policy_rows.extend(policies)

    if not rows:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No metric JSON files found under: {searched}")

    df = pd.DataFrame(rows)
    policy_df = pd.DataFrame(policy_rows)
    main_df = ordered_main_results(df)
    if main_df.empty:
        raise FileNotFoundError("No baseline/resnet50/torchxrayvision result directories found.")

    save_internal_external_auc(main_df, out_dir)
    save_external_metric_comparison(main_df, out_dir)
    save_auc_drop(main_df, out_dir)
    save_threshold_policy_plot(policy_df, out_dir)
    save_ablation_plot(df, out_dir)
    write_manifest(df, policy_df, out_dir)

    print(f"Saved figures to: {out_dir}")
    print(f"Result dirs found: {len(df)}")
    print(f"Prediction CSV pairs found: {int(df['has_predictions'].sum())}")
    print(f"Threshold JSON files found: {int(df['has_thresholds'].sum())}")


if __name__ == "__main__":
    main()
