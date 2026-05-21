#!/bin/bash
# evaluate_all_models.py sbatch 제출 래퍼 (KHU Seraph / moana)
#
# 사용법:
#   bash scripts/run_eval.sh baseline_cnn     <ckpt>
#   bash scripts/run_eval.sh resnet50         <ckpt>
#   bash scripts/run_eval.sh torchxrayvision  <ckpt>
#
# 실제 잡은 scripts/eval_job.sh (sbatch 스크립트).
# 마스터 노드에서는 sbatch 제출만 하고, 평가/unzip/python 은 모두 compute node 에서 수행.

set -e

MODEL="${1:?사용법: $0 <model> <checkpoint>}"
CHECKPOINT="${2:?사용법: $0 <model> <checkpoint>}"
OUT_TAG="${3:-${MODEL}_multi}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

if [ ! -f "$CHECKPOINT" ]; then
  echo "[ERROR] checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

echo "Submitting eval job"
echo "  MODEL      : $MODEL"
echo "  CHECKPOINT : $CHECKPOINT"
echo "  OUT_TAG    : $OUT_TAG"

sbatch \
  --job-name="eval_${MODEL}" \
  --export=ALL,MODEL="$MODEL",CHECKPOINT="$CHECKPOINT",OUT_TAG="$OUT_TAG" \
  "$REPO_DIR/scripts/eval_job.sh"

echo "제출 완료. 로그: $LOG_DIR/<JOBID>_eval.out"
echo "상태 확인: squeue -u $USER"
echo "         : sacct -j <JOBID> --format=JobID,JobName,State,Elapsed"
