#!/bin/bash
#SBATCH --job-name=phase2_full_beauty_var_resid
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/full_beauty_var_resid.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/full_beauty_var_resid.err

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

CHECKPOINT="saved/ContGCN-Jun-25-2026_15-19-14.pth"
DATASET="amazon-beauty"
PREFIX="full_beauty_var_resid"

echo "===================================================="
echo "Phase 2 residual connector eval"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Prefix: ${PREFIX}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python eval_phase2_removed_items.py \
  --checkpoint "${CHECKPOINT}" \
  --cold_dataset "${DATASET}" \
  --modes random,direct,connector \
  --out_dir logs_connector/phase2 \
  --prefix "${PREFIX}"

echo "Finished Phase 2 at: $(date)"
