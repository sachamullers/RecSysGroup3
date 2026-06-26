#!/bin/bash
#SBATCH --job-name=full_beauty_var_resid
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/full_beauty_var_resid.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/full_beauty_var_resid.err

cd /gpfs/home5/scur1242/RecSysGroup3

mkdir -p logs_connector
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

echo "===================================================="
echo "Experiment: full_beauty_var_resid"
echo "Dataset: amazon-beauty"
echo "Model: ContGCN"
echo "Opt: var"
echo "Residual connector alpha: 0.1"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "===================================================="

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

python run_recbole.py \
  -d amazon-beauty \
  -m ContGCN \
  --opt var 

echo "Finished training at: $(date)"
