#!/bin/bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: $0 [checkpoint.pt] [split_csv] [out_dir]" >&2
  echo "Runs inference only. Use this inside an srun GPU allocation on SERAPH." >&2
  exit 0
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${1:-$REPO_DIR/outputs/torchxrayvision/best_torchxrayvision_seed42.pt}"
SPLIT_CSV="${2:-$REPO_DIR/outputs/splits/kaggle_split_seed42.csv}"
OUT_DIR="${3:-$REPO_DIR/outputs/torchxrayvision_external}"
DATA_ROOT="${DATA_ROOT:-/local_datasets/$USER/chest_xray_kaggle/chest_xray}"
THRESHOLD_JSON="${THRESHOLD_JSON:-$OUT_DIR/internal_threshold_seed42.json}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

cd "$REPO_DIR"
python src/export_kaggle_val_predictions.py \
  --checkpoint "$CHECKPOINT" \
  --split_csv "$SPLIT_CSV" \
  --threshold_json "$THRESHOLD_JSON" \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --image_size "$IMAGE_SIZE" \
  --seed "$SEED" \
  --device "$DEVICE"
