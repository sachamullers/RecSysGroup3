#!/bin/bash
#SBATCH --job-name=block_sgl
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/scur1242/RecSysGroup3/logsfinal/%x-%A_%a.out
#SBATCH --error=/home/scur1242/RecSysGroup3/logsfinal/%x-%A_%a.err

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    module load 2024
    module load Anaconda3/2024.06-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate llminit

echo "Python: $(which python)"
python -c "import sys; print(sys.executable)"
python -c "import ray; print('ray ok')"

cd /projects/prjs2120/groups/group_03/code/RecSysGroup3

COMMANDS=(
  "python run_recbole.py -d amazon-tools-home-improvement -m SGL"
  "python run_recbole.py --opt rand -d amazon-tools-home-improvement -m ContSGL"
  "python run_recbole.py --opt uni  -d amazon-tools-home-improvement -m ContSGL"
  "python run_recbole.py --opt var  -d amazon-tools-home-improvement -m ContSGL"

  "python run_recbole.py -d amazon-office-products -m SGL"
  "python run_recbole.py --opt rand -d amazon-office-products -m ContSGL"
  "python run_recbole.py --opt uni  -d amazon-office-products -m ContSGL"
  "python run_recbole.py --opt var  -d amazon-office-products -m ContSGL"
)

CMD="${COMMANDS[$SLURM_ARRAY_TASK_ID]}"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Running on: $SLURMD_NODENAME"
echo "Command: $CMD"
nvidia-smi
date

eval "$CMD"

date
echo "Done."
