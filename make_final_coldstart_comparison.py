from pathlib import Path
import pandas as pd


out_dir = Path("logs_connector/phase2")
out_dir.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "method": "Random lower bound",
        "model_group": "author_baseline",
        "checkpoint": "saved/ContGCN-Jun-25-2026_15-19-14.pth",
        "training_setup": "ContGCN-var, use_connector=False",
        "phase2_mode": "random",
        "cold_item_embedding": "random 128-dim vector",
        "num_eval_users": 3846,
        "removed_only_recall@10": 0.0157,
        "removed_only_ndcg@10": 0.0069,
        "removed_only_recall@20": 0.0330,
        "removed_only_ndcg@20": 0.0118,
        "removed_only_recall@50": 0.0833,
        "removed_only_ndcg@50": 0.0232,
        "seen_plus_removed_recall@10": 0.0003,
        "seen_plus_removed_ndcg@10": 0.0001,
        "seen_plus_removed_recall@20": 0.0004,
        "seen_plus_removed_ndcg@20": 0.0001,
        "seen_plus_removed_recall@50": 0.0007,
        "seen_plus_removed_ndcg@50": 0.0002,
    },
    {
        "method": "Authors LLMInit",
        "model_group": "author_baseline",
        "checkpoint": "saved/ContGCN-Jun-25-2026_15-19-14.pth",
        "training_setup": "ContGCN-var, use_connector=False",
        "phase2_mode": "direct",
        "cold_item_embedding": "direct LLMInit-var selected dimensions",
        "num_eval_users": 3846,
        "removed_only_recall@10": 0.1985,
        "removed_only_ndcg@10": 0.1252,
        "removed_only_recall@20": 0.2912,
        "removed_only_ndcg@20": 0.1523,
        "removed_only_recall@50": 0.4229,
        "removed_only_ndcg@50": 0.1830,
        "seen_plus_removed_recall@10": 0.0000,
        "seen_plus_removed_ndcg@10": 0.0000,
        "seen_plus_removed_recall@20": 0.0000,
        "seen_plus_removed_ndcg@20": 0.0000,
        "seen_plus_removed_recall@50": 0.0000,
        "seen_plus_removed_ndcg@50": 0.0000,
    },
    {
        "method": "Ours: plain connector",
        "model_group": "ours_plain_connector",
        "checkpoint": "plain connector checkpoint",
        "training_setup": "ContGCN-var, use_connector=True, residual=False",
        "phase2_mode": "connector",
        "cold_item_embedding": "connector(full 768-dim LLM item embedding)",
        "num_eval_users": 3846,
        "removed_only_recall@10": 0.107755,
        "removed_only_ndcg@10": 0.059178,
        "removed_only_recall@20": None,
        "removed_only_ndcg@20": None,
        "removed_only_recall@50": None,
        "removed_only_ndcg@50": None,
        "seen_plus_removed_recall@10": 0.000312,
        "seen_plus_removed_ndcg@10": 0.000257,
        "seen_plus_removed_recall@20": None,
        "seen_plus_removed_ndcg@20": None,
        "seen_plus_removed_recall@50": None,
        "seen_plus_removed_ndcg@50": None,
    },
    {
        "method": "Ours: residual connector alpha=0.1",
        "model_group": "ours_residual_connector",
        "checkpoint": "residual connector checkpoint",
        "training_setup": "ContGCN-var, use_connector=True, residual=True, alpha=0.1",
        "phase2_mode": "connector",
        "cold_item_embedding": "direct LLMInit-var + 0.1 * connector(full 768-dim LLM item embedding)",
        "num_eval_users": 3846,
        "removed_only_recall@10": 0.139537,
        "removed_only_ndcg@10": 0.084717,
        "removed_only_recall@20": None,
        "removed_only_ndcg@20": None,
        "removed_only_recall@50": None,
        "removed_only_ndcg@50": None,
        "seen_plus_removed_recall@10": 0.012665,
        "seen_plus_removed_ndcg@10": 0.005355,
        "seen_plus_removed_recall@20": None,
        "seen_plus_removed_ndcg@20": None,
        "seen_plus_removed_recall@50": None,
        "seen_plus_removed_ndcg@50": None,
    },
]

df = pd.DataFrame(rows)

author_direct = df[df["method"] == "Authors LLMInit"].iloc[0]
plain_connector = df[df["method"] == "Ours: plain connector"].iloc[0]

df["delta_removed_only_ndcg@10_vs_authors_llminit"] = (
    df["removed_only_ndcg@10"] - author_direct["removed_only_ndcg@10"]
)
df["delta_seen_plus_removed_ndcg@10_vs_authors_llminit"] = (
    df["seen_plus_removed_ndcg@10"] - author_direct["seen_plus_removed_ndcg@10"]
)

df["delta_removed_only_ndcg@10_vs_plain_connector"] = None
df["delta_seen_plus_removed_ndcg@10_vs_plain_connector"] = None

connector_mask = df["method"].str.contains("connector", case=False, na=False)

df.loc[connector_mask, "delta_removed_only_ndcg@10_vs_plain_connector"] = (
    df.loc[connector_mask, "removed_only_ndcg@10"] - plain_connector["removed_only_ndcg@10"]
)
df.loc[connector_mask, "delta_seen_plus_removed_ndcg@10_vs_plain_connector"] = (
    df.loc[connector_mask, "seen_plus_removed_ndcg@10"] - plain_connector["seen_plus_removed_ndcg@10"]
)

out_path = out_dir / "beauty_core_final_coldstart_comparison.csv"
df.to_csv(out_path, index=False)

print(df.to_string(index=False))
print(f"\nWrote: {out_path}")