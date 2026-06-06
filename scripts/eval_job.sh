#!/bin/bash
#SBATCH -J eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 2:00:00
#SBATCH -p batch_ce_ugrad
#SBATCH -o /data/woohyuk/aip_project/logs/%j_eval.out

set -euo pipefail

# Required env vars (set via sbatch --export):
#   MODEL      - baseline_cnn | resnet50 | torchxrayvision
#   CHECKPOINT - absolute path to .pt
#   OUT_TAG    - subdir name under outputs/ (optional, default ${MODEL}_multi)

MODEL="${MODEL:?MODEL env var required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT env var required}"
OUT_TAG="${OUT_TAG:-${MODEL}_multi}"

REPO_DIR="/data/woohyuk/aip_project"
NAS_CACHE="/data/woohyuk/aip_data"
LOCAL_DIR="/local_datasets/woohyuk/aip_data"

echo "=== Job info ==="
hostname; date
echo "MODEL=$MODEL  CHECKPOINT=$CHECKPOINT  OUT_TAG=$OUT_TAG"

echo "=== Step 1: copy NAS cache -> node local ==="
mkdir -p "$LOCAL_DIR"
time rsync -a "$NAS_CACHE/chest_xray/" "$LOCAL_DIR/chest_xray/"
time rsync -a "$NAS_CACHE/rsna/"        "$LOCAL_DIR/rsna/"
du -sh "$LOCAL_DIR/chest_xray" "$LOCAL_DIR/rsna"

echo "=== Step 2: activate conda env ==="
source /data/woohyuk/anaconda3/etc/profile.d/conda.sh
conda activate aip
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== Step 3: prepare_kaggle_split.py (local paths) ==="
KAGGLE_CSV="$LOCAL_DIR/kaggle_split.csv"
cd "$REPO_DIR"
python src/prepare_kaggle_split.py \
    --data_root "$LOCAL_DIR/chest_xray/chest_xray" \
    --out_csv   "$KAGGLE_CSV"
head -3 "$KAGGLE_CSV"

echo "=== Step 4: evaluate_all_models.py ==="
OUT_DIR="$REPO_DIR/outputs/$OUT_TAG"
mkdir -p "$OUT_DIR"
python src/evaluate_all_models.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --kaggle_csv "$KAGGLE_CSV" \
    --rsna_root  "$LOCAL_DIR/rsna" \
    --out_dir    "$OUT_DIR" \
    --n_bootstrap 1000

echo "=== Done ==="
date
exit 0
