#!/bin/bash
#SBATCH --job-name=recon
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --partition=debug_ce_ugrad
#SBATCH -o /home/woohyuk/aip_project/logs/%j_recon.out

echo "=== hostname / whoami / date ==="
hostname; whoami; date

echo "=== /local_datasets ==="
ls -la /local_datasets/ 2>&1 | head -30

echo "=== /local_datasets/woohyuk ==="
ls -la /local_datasets/woohyuk/ 2>&1

echo "=== mkdir test ==="
mkdir -p /local_datasets/woohyuk/test_recon 2>&1 && echo "mkdir OK" && rmdir /local_datasets/woohyuk/test_recon

echo "=== df -h /local_datasets ==="
df -h /local_datasets 2>&1

echo "=== /data/woohyuk listing ==="
ls /data/woohyuk/ | head -10

echo "=== /data/daniel3290/datasets/tarfiles ==="
ls -la /data/daniel3290/datasets/tarfiles/ 2>&1

echo "=== conda activate aip + torch ==="
source /data/woohyuk/anaconda3/etc/profile.d/conda.sh
conda activate aip
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

exit 0
