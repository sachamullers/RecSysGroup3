#!/bin/bash
#SBATCH --job-name=topk_low5_var
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/topk_beauty_low5_var.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/phase2/topk_beauty_low5_var.err

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

CHECKPOINT="saved/REPLACE_WITH_LOW5_CHECKPOINT.pth"
DATASET="amazon-beauty-low5cold"
OUT_CSV="logs_connector/phase2/beauty_low5_contgcn_var_topk_examples.csv"

echo "Exporting top-k examples"
echo "Dataset: ${DATASET}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_CSV}"
echo "Time: $(date)"
echo "Node: $(hostname)"

python export_phase2_topk_examples.py \
  --checkpoint "${CHECKPOINT}" \
  --cold_dataset "${DATASET}" \
  --modes random,direct,connector \
  --topk 10 \
  --num_users 100 \
  --out "${OUT_CSV}"

echo "Finished at: $(date)"
