#!/bin/bash
#SBATCH --job-name=connector_sgl
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-5%2
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/slurm_%x-%A_%a.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/slurm_%x-%A_%a.err

cd /gpfs/home5/scur1242/RecSysGroup3

mkdir -p logs_connector

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    module load 2024
    module load Anaconda3/2024.06-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate llminit

case ${SLURM_ARRAY_TASK_ID} in
    0)
        DATASET="amazon-office-products"
        OPT="rand"
        ;;
    1)
        DATASET="amazon-office-products"
        OPT="uni"
        ;;
    2)
        DATASET="amazon-office-products"
        OPT="var"
        ;;
    3)
        DATASET="amazon-tools-home-improvement"
        OPT="rand"
        ;;
    4)
        DATASET="amazon-tools-home-improvement"
        OPT="uni"
        ;;
    5)
        DATASET="amazon-tools-home-improvement"
        OPT="var"
        ;;
    *)
        echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

MODEL="ContSGL"
RUN_NAME="connector_${MODEL}_${DATASET}_${OPT}"

OUT_LOG="logs_connector/${RUN_NAME}.out"
ERR_LOG="logs_connector/${RUN_NAME}.err"

exec > "${OUT_LOG}" 2> "${ERR_LOG}"

echo "=========================================="
echo "Run name: ${RUN_NAME}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Option: ${OPT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "=========================================="

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

echo "Starting RecBole run..."
echo "Command: python run_recbole.py -d ${DATASET} -m ${MODEL} --opt ${OPT}"

python run_recbole.py -d "${DATASET}" -m "${MODEL}" --opt "${OPT}"

echo "Finished at: $(date)"
