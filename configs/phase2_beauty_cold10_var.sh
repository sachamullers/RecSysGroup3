#!/bin/bash
#SBATCH --job-name=phase2_beauty_var
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2_%x-%j.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2_%x-%j.err

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

python eval_phase2_removed_items.py \
  --checkpoint saved/ContGCN-Jun-21-2026_21-44-52.pth \
  --cold_dataset amazon-beauty-cold10 \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_cold10_contgcn_var