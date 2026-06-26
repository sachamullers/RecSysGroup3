#!/bin/bash
#SBATCH --job-name=beauty_core_var
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/beauty_core_var.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/beauty_core_var.err

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

echo "Dataset: amazon-beauty-cold10core"
echo "Model: ContGCN"
echo "Opt: var"
echo "Time: $(date)"
echo "Node: $(hostname)"

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python run_recbole.py -d amazon-beauty-cold10core -m ContGCN --opt var
