#!/bin/bash
#SBATCH -J pip-pydicom
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 0:05:00
#SBATCH -p debug_ce_ugrad
#SBATCH -o /data/woohyuk/aip_project/logs/%j_install_pydicom.out

set -euo pipefail
hostname; date
source /data/woohyuk/anaconda3/etc/profile.d/conda.sh
conda activate aip
pip install pydicom
python -c "import pydicom; print('pydicom', pydicom.__version__)"
exit 0
