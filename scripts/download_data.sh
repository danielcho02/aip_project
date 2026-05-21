#!/bin/bash
# Kaggle 데이터셋 다운로드 스크립트
# 사용법: bash scripts/download_data.sh
#
# 사전 준비:
#   ~/.kaggle/kaggle.json 에 API 키 저장되어 있어야 함
#   (https://www.kaggle.com/settings > API > Create New Token)

set -e

DATA_DIR="/data/$USER/aip_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[1/4] Kaggle chest X-ray pneumonia 데이터셋 다운로드..."
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -q chest-xray-pneumonia.zip -d chest_xray
rm chest-xray-pneumonia.zip
echo "  -> $DATA_DIR/chest_xray"

echo "[2/4] RSNA pneumonia detection 데이터셋 다운로드..."
# join the competition 먼저 해야 함: https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip -q rsna-pneumonia-detection-challenge.zip -d rsna
rm rsna-pneumonia-detection-challenge.zip
echo "  -> $DATA_DIR/rsna"

echo "[3/4] kaggle split CSV 생성..."
cd -
python src/prepare_kaggle_split.py \
    --data_root "$DATA_DIR/chest_xray/chest_xray" \
    --out_csv "$DATA_DIR/kaggle_split.csv"
echo "  -> $DATA_DIR/kaggle_split.csv"

echo "[4/4] 데이터 확인..."
python - <<EOF
import pandas as pd
df = pd.read_csv("$DATA_DIR/kaggle_split.csv")
print(df.groupby(["split", "class_name"]).size().to_string())
EOF

echo ""
echo "데이터 준비 완료."
echo "  KAGGLE_CSV : $DATA_DIR/kaggle_split.csv"
echo "  RSNA_ROOT  : $DATA_DIR/rsna"
