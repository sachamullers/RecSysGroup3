#!/bin/bash
#SBATCH --job-name=beauty_low5_var_noresid_run2
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/low5/beauty_low5_var_noresid_run2.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/low5/beauty_low5_var_noresid_run2.err

cd /gpfs/home5/scur1242/RecSysGroup3

mkdir -p logs_connector/low5
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

DATASET="amazon-beauty-low5cold"
MODEL="ContGCN"
OPT="var"
RUN_NAME="beauty_low5_contgcn_var_noresid_run2"

echo "===================================================="
echo "Experiment: ${RUN_NAME}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Opt: ${OPT}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python run_recbole.py -d "${DATASET}" -m "${MODEL}" --opt "${OPT}"

echo "Finished training at: $(date)"
