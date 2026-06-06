#!/bin/bash
#SBATCH -J aip-data-prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH -p debug_ce_ugrad
#SBATCH -o /data/woohyuk/aip_project/logs/%j_data_prep.out

set -euo pipefail

NAS_ZIP_DIR="/data/daniel3290/datasets/tarfiles"
LOCAL_DIR="/local_datasets/woohyuk/aip_data"
NAS_CACHE="/data/woohyuk/aip_data"

echo "=== Host / Time ==="
hostname; date

echo "=== Step 1: prepare local + NAS cache dirs ==="
mkdir -p "$LOCAL_DIR" "$NAS_CACHE"
df -h "$LOCAL_DIR"

echo "=== Step 2: unzip chest-xray-pneumonia.zip -> local ==="
time unzip -q -o "$NAS_ZIP_DIR/chest-xray-pneumonia.zip" -d "$LOCAL_DIR/chest_xray"
ls "$LOCAL_DIR/chest_xray/" | head -10

echo "=== Step 3: unzip rsna-pneumonia-detection-challenge.zip -> local ==="
time unzip -q -o "$NAS_ZIP_DIR/rsna-pneumonia-detection-challenge.zip" -d "$LOCAL_DIR/rsna"
ls "$LOCAL_DIR/rsna/" | head -10

echo "=== Step 4: copy extracted dirs back to NAS cache (one-time) ==="
time rsync -a --delete "$LOCAL_DIR/chest_xray/" "$NAS_CACHE/chest_xray/"
time rsync -a --delete "$LOCAL_DIR/rsna/"        "$NAS_CACHE/rsna/"

echo "=== Step 5: verify NAS cache ==="
du -sh "$NAS_CACHE/chest_xray" "$NAS_CACHE/rsna"
ls "$NAS_CACHE/chest_xray/chest_xray/" 2>&1 | head -10 || true
ls "$NAS_CACHE/chest_xray/" 2>&1 | head -10

echo "=== Done ==="
date
exit 0
