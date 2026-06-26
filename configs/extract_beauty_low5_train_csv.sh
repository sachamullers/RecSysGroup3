#!/bin/bash

cd /gpfs/home5/scur1242/RecSysGroup3

python - <<'PY'
import re
import csv
from pathlib import Path

err_path = Path("logs_connector/low5/beauty_low5_var.err")
out_csv = Path("logs_connector/low5/beauty_low5_train_metrics.csv")

text = err_path.read_text(errors="ignore")

checkpoint_match = re.findall(r"Loading model structure and parameters from\s+([^\s]+\.pth)", text)
checkpoint = checkpoint_match[-1] if checkpoint_match else ""

test_match = re.findall(r"test result:\s*OrderedDict\(\[(.*?)\]\)", text, re.S)

metrics = {}
if test_match:
    pairs = re.findall(r"\('([^']+)',\s*([0-9.eE+-]+)\)", test_match[-1])
    metrics = {k: v for k, v in pairs}

row = {
    "experiment": "beauty_low5_contgcn_var",
    "setting": "pure_cold_low5",
    "dataset": "amazon-beauty-low5cold",
    "base_dataset": "amazon-beauty",
    "model": "ContGCN",
    "opt": "var",
    "checkpoint": checkpoint,
    "test_recall@10": metrics.get("recall@10", ""),
    "test_recall@20": metrics.get("recall@20", ""),
    "test_recall@50": metrics.get("recall@50", ""),
    "test_ndcg@10": metrics.get("ndcg@10", ""),
    "test_ndcg@20": metrics.get("ndcg@20", ""),
    "test_ndcg@50": metrics.get("ndcg@50", ""),
    "test_mrr@10": metrics.get("mrr@10", ""),
    "test_hit@10": metrics.get("hit@10", ""),
    "test_precision@10": metrics.get("precision@10", ""),
    "log_file": str(err_path),
}

out_csv.parent.mkdir(parents=True, exist_ok=True)

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)

print("Wrote:", out_csv)
print("Checkpoint:", checkpoint)
print("R@10:", row["test_recall@10"])
print("N@10:", row["test_ndcg@10"])
PY
