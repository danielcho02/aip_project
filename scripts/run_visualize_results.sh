#!/bin/bash
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  echo "Usage: $0 [outputs_root] [external_outputs_root] [out_dir]" >&2
  echo "CPU-only result plotting; safe to run on the master/login node." >&2
  exit 0
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUTS_ROOT="${1:-$REPO_DIR/outputs}"
EXTERNAL_OUTPUTS_ROOT="${2:-$REPO_DIR/external_outputs}"
OUT_DIR="${3:-$REPO_DIR/outputs/figures}"

cd "$REPO_DIR"
python src/visualize_results.py \
  --outputs_root "$OUTPUTS_ROOT" \
  --external_outputs_root "$EXTERNAL_OUTPUTS_ROOT" \
  --out_dir "$OUT_DIR"
