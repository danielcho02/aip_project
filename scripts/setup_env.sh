#!/bin/bash
# Seraph (moana) conda 환경 세팅 스크립트
# 사용법: bash scripts/setup_env.sh

set -e

CONDA_BIN="/data/$USER/anaconda3/bin/conda"
ENV_NAME="aip"
PYTHON_VERSION="3.10"

echo "[1/4] conda 초기화..."
$CONDA_BIN init bash
source ~/.bashrc

echo "[2/4] 환경 생성 (${ENV_NAME}, Python ${PYTHON_VERSION})..."
$CONDA_BIN create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "[3/4] 패키지 설치..."
source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchxrayvision timm scikit-learn pandas pillow tqdm kaggle

echo "[4/4] 설치 확인..."
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import torchxrayvision; print('torchxrayvision OK')"
python -c "import timm; print('timm OK')"

echo ""
echo "완료. 이후 매번: source /data/$USER/anaconda3/etc/profile.d/conda.sh && conda activate $ENV_NAME"
