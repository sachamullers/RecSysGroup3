#!/bin/bash
#SBATCH --job-name=phase2_low5_var_noresid_run2
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/beauty_low5_var_noresid_run2.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/beauty_low5_var_noresid_run2.err

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

CHECKPOINT="saved/ContGCN-Jun-26-2026_19-41-16.pth"
DATASET="amazon-beauty-low5cold"
PREFIX="beauty_low5_contgcn_var_noresid_run2"

echo "===================================================="
echo "Phase 2 pure cold-start eval"
echo "Prefix: ${PREFIX}"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
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
