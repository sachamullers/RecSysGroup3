#!/bin/bash
#SBATCH --job-name=phase2_beauty_core_align
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2_beauty_core_align.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2_beauty_core_align.err

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
  --checkpoint saved/ContGCN-Jun-24-2026_23-21-11.pth \
  --cold_dataset amazon-beauty-cold10core \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix beauty_core_contgcn_var_align001