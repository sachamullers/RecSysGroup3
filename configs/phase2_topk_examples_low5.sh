#!/bin/bash
#SBATCH --job-name=phase2_topk_examples
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/beauty_low5_topk_examples.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/beauty_low5_topk_examples.err

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

CHECKPOINT="saved/ContGCN-Jun-25-2026_10-29-58.pth"
DATASET="amazon-beauty-low5cold"
OUT_FILE="logs_connector/phase2/beauty_low5_contgcn_var_topk_examples.csv"

echo "===================================================="
echo "Phase 2 top-k examples export"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_FILE}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

python export_phase2_topk_examples.py --checkpoint "${CHECKPOINT}" --cold_dataset "${DATASET}" --modes random,direct,connector --topk 10 --num_users 100 --out "${OUT_FILE}"

echo "Finished Phase 2 top-k examples export at: $(date)"
