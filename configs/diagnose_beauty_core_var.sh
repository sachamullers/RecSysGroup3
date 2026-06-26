#!/bin/bash
#SBATCH --job-name=diag_beauty_var
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/diag_beauty_var.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/diag_beauty_var.err

cd /gpfs/home5/scur1242/RecSysGroup3

mkdir -p logs_connector/phase2

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    module load 2024
    module load Anaconda3/2024.06-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate llminit

echo "Running connector scale diagnostic"
echo "Dataset: amazon-beauty-cold10core"
echo "Checkpoint: saved/ContGCN-Jun-24-2026_22-05-11.pth"
echo "Output CSV: logs_connector/phase2/beauty_core_contgcn_var_diagnostic_scale.csv"
echo "Time: $(date)"
echo "Node: $(hostname)"

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python diagnose_connector_scale.py \
  --checkpoint saved/ContGCN-Jun-24-2026_22-05-11.pth \
  --cold_dataset amazon-beauty-cold10core \
  --out logs_connector/phase2/beauty_core_contgcn_var_diagnostic_scale.csv
