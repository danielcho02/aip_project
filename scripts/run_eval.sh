#!/bin/bash
# evaluate_all_models.py sbatch 실행 래퍼
#
# 사용법:
#   bash scripts/run_eval.sh resnet50   <checkpoint_path>
#   bash scripts/run_eval.sh torchxrayvision <checkpoint_path>
#   bash scripts/run_eval.sh baseline_cnn   <checkpoint_path>
#
# 예시:
#   bash scripts/run_eval.sh resnet50 /data/woohyuk/aip_checkpoints/resnet50_best.pt

set -e

MODEL="${1:?사용법: $0 <model> <checkpoint>}"
CHECKPOINT="${2:?사용법: $0 <model> <checkpoint>}"

DATA_DIR="/data/$USER/aip_data"
KAGGLE_CSV="$DATA_DIR/kaggle_split.csv"
RSNA_ROOT="$DATA_DIR/rsna"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"

CONDA_ACTIVATE="source /data/$USER/anaconda3/etc/profile.d/conda.sh && conda activate aip"

CMD="$CONDA_ACTIVATE && cd $REPO_DIR && python src/evaluate_all_models.py \
  --model $MODEL \
  --checkpoint $CHECKPOINT \
  --kaggle_csv $KAGGLE_CSV \
  --rsna_root $RSNA_ROOT \
  --out_dir outputs/${MODEL}_multi \
  --n_bootstrap 1000"

echo "Submitting: $MODEL"
echo "Checkpoint: $CHECKPOINT"

sbatch \
  --job-name="eval_${MODEL}" \
  --gres=gpu:1 \
  --cpus-per-gpu=8 \
  --mem-per-gpu=32G \
  --time=2:00:00 \
  --partition=batch_ugrad \
  -o "$LOG_DIR/%j_eval_${MODEL}.out" \
  --wrap="$CMD"

echo "제출 완료. 로그: $LOG_DIR/<JOBID>_eval_${MODEL}.out"
echo "상태 확인: sacct -j <JOBID> --format=JobID,JobName,State,Elapsed"
