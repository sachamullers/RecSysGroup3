#!/bin/bash
#SBATCH --job-name=fair_plain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/fairness_plain_%j.out
#SBATCH --error=/gpfs/home5/scur1242/RecSysGroup3/logs_connector/fairness_plain_%j.err

cd /gpfs/home5/scur1242/RecSysGroup3

source ~/.bashrc

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    module load 2024
    module load Anaconda3/2024.06-1
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate llminit

mkdir -p data_analysis/fairness_eval/recommendations
mkdir -p data_analysis/fairness_eval/outputs
mkdir -p logs_connector

REC_OUT="data_analysis/fairness_eval/recommendations/beauty_plain_connector_top20.csv"
OUT_DIR="data_analysis/fairness_eval/outputs/beauty_plain_connector_b100_k10"
ITEMS="dataset/amazon-beauty/amazon-beauty-without-emb.item"

python data_analysis/fairness_eval/export_topk_recommendations.py \
  --model-file saved/ContGCN-Jun-21-2026_21-52-04.pth \
  --output "${REC_OUT}" \
  --k 20 \
  --batch-size 128 \
  --external-ids

python data_analysis/fairness_eval/fairness_metrics_posthoc.py \
  --recommendations "${REC_OUT}" \
  --interactions dataset/amazon-beauty/amazon-beauty.inter \
  --items "${ITEMS}" \
  --k 10 \
  --tail-buckets 100 \
  --model-name "Plain connector ContGCN" \
  --output-dir "${OUT_DIR}" \
  --no-plots

python - <<'PY'
import pandas as pd
from pathlib import Path

out_dir = Path("data_analysis/fairness_eval/outputs/beauty_plain_connector_b100_k10")
summary = pd.read_csv(out_dir / "brand_metrics_summary_at_10.csv").iloc[0]

print("\nFinal plain connector fairness values:")
print("Coverage@10:", f"{summary['catalog_coverage@10']:.4f}")
print("Brand Div.@10:", f"{summary['avg_brand_diversity@10']:.4f}")
print("Inv. Pop. Exposure@10:", f"{summary['avg_tail_item_diversity@10']:.4f}")
print("Brand-loyal:", f"{summary['brand_loyal_avg_diversity@10']:.4f}")
print("Non-brand-loyal:", f"{summary['non_brand_loyal_avg_diversity@10']:.4f}")
print("Gap:", f"{summary['brand_diversity_gap_loyal_minus_non_loyal@10']:.4f}")
PY