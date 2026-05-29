#!/bin/bash
#SBATCH -J gradcam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 1:00:00
#SBATCH -p batch_ce_ugrad
#SBATCH -o logs/%j_gradcam.out

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR env var required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT env var required}"
RESULT_DIR="${RESULT_DIR:?RESULT_DIR env var required}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/figures/gradcam}"
KAGGLE_ROOT="${KAGGLE_ROOT:-/local_datasets/$USER/chest_xray_kaggle/chest_xray}"
RSNA_ROOT="${RSNA_ROOT:-/local_datasets/$USER/rsna}"
SYNC_DATA="${SYNC_DATA:-0}"
CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-aip_project}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SAMPLES_PER_CASE="${SAMPLES_PER_CASE:-2}"

echo "=== Job info ==="
hostname
date
echo "REPO_DIR=$REPO_DIR"
echo "CHECKPOINT=$CHECKPOINT"
echo "RESULT_DIR=$RESULT_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "KAGGLE_ROOT=$KAGGLE_ROOT"
echo "RSNA_ROOT=$RSNA_ROOT"
echo "SYNC_DATA=$SYNC_DATA"

if [ "$SYNC_DATA" = "1" ]; then
  NAS_KAGGLE_ROOT="${NAS_KAGGLE_ROOT:?NAS_KAGGLE_ROOT env var required when SYNC_DATA=1}"
  NAS_RSNA_ROOT="${NAS_RSNA_ROOT:?NAS_RSNA_ROOT env var required when SYNC_DATA=1}"
  echo "=== Sync data to node-local dataset directories ==="
  mkdir -p "$KAGGLE_ROOT" "$RSNA_ROOT"
  rsync -a "$NAS_KAGGLE_ROOT/" "$KAGGLE_ROOT/"
  rsync -a "$NAS_RSNA_ROOT/" "$RSNA_ROOT/"
else
  echo "=== Using existing node-local dataset directories; no rsync ==="
fi

echo "=== Activate conda env ==="
source "$CONDA_SH"
conda activate "$CONDA_ENV"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd "$REPO_DIR"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --kaggle_predictions "$RESULT_DIR/kaggle_val_predictions_seed42.csv"
  --rsna_predictions "$RESULT_DIR/rsna_predictions_seed42.csv"
  --threshold_json "$RESULT_DIR/threshold_policies_seed42.json"
  --out_dir "$OUT_DIR"
  --kaggle_root "$KAGGLE_ROOT"
  --rsna_root "$RSNA_ROOT"
  --image_size "$IMAGE_SIZE"
  --samples_per_case "$SAMPLES_PER_CASE"
  --device cuda
)

if [ -n "${PRED_PATH_PREFIX_FROM:-}" ] && [ -n "${PRED_PATH_PREFIX_TO:-}" ]; then
  ARGS+=(--path_prefix_from "$PRED_PATH_PREFIX_FROM" --path_prefix_to "$PRED_PATH_PREFIX_TO")
fi

if [ "${BODY_CROP:-1}" = "0" ]; then
  ARGS+=(--no_body_crop)
fi

python src/visualize_gradcam.py "${ARGS[@]}"

echo "=== Done ==="
date
