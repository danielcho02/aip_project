#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <checkpoint.pt> <result_dir> [out_dir]" >&2
  echo "result_dir must contain kaggle_val_predictions_seed42.csv, rsna_predictions_seed42.csv, and threshold_policies_seed42.json." >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_INPUT="$1"
RESULT_DIR_INPUT="$2"
OUT_DIR="${3:-$REPO_DIR/outputs/figures/gradcam}"
LOG_DIR="$REPO_DIR/logs"
CONDA_ENV="${CONDA_ENV:-aip_project}"
mkdir -p "$LOG_DIR"

if [ ! -f "$CHECKPOINT_INPUT" ]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT_INPUT" >&2
  exit 2
fi
if [ ! -d "$RESULT_DIR_INPUT" ]; then
  echo "[ERROR] result_dir not found: $RESULT_DIR_INPUT" >&2
  exit 2
fi

CHECKPOINT="$(realpath "$CHECKPOINT_INPUT")"
RESULT_DIR="$(realpath "$RESULT_DIR_INPUT")"

cd "$REPO_DIR"
sbatch \
  --job-name="gradcam_txrv" \
  --export=ALL,REPO_DIR="$REPO_DIR",CHECKPOINT="$CHECKPOINT",RESULT_DIR="$RESULT_DIR",OUT_DIR="$OUT_DIR",KAGGLE_ROOT="${KAGGLE_ROOT:-}",RSNA_ROOT="${RSNA_ROOT:-}",SYNC_DATA="${SYNC_DATA:-}",NAS_KAGGLE_ROOT="${NAS_KAGGLE_ROOT:-}",NAS_RSNA_ROOT="${NAS_RSNA_ROOT:-}",CONDA_ENV="$CONDA_ENV",PRED_PATH_PREFIX_FROM="${PRED_PATH_PREFIX_FROM:-}",PRED_PATH_PREFIX_TO="${PRED_PATH_PREFIX_TO:-}" \
  "$REPO_DIR/scripts/gradcam_job.sh"

echo "Submitted Grad-CAM job. Logs: $LOG_DIR/<JOBID>_gradcam.out"
