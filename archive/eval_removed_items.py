import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from recbole.quick_start.quick_start import load_data_and_model


def find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"Could not find column starting with {prefix}. Columns: {list(df.columns)}")
    return matches[0]


def parse_float_seq(x):
    return np.array([float(v) for v in str(x).split()], dtype=np.float32)


def build_token_map(dataset, field):
    """
    RecBole usually stores field2token_id[field].
    Depending on version, this can be dict-like or list-like.
    We convert it to token -> inner_id.
    """
    token_info = dataset.field2token_id[field]

    if isinstance(token_info, dict):
        return {str(k): int(v) for k, v in token_info.items()}

    # list/np array where index = inner id, value = raw token
    return {str(tok): int(i) for i, tok in enumerate(token_info)}


def recall_ndcg_from_ranks(ranked_indices, relevant_indices, k):
    relevant = set(relevant_indices)
    topk = ranked_indices[:k]

    hits = [1 if idx in relevant else 0 for idx in topk]
    recall = sum(hits) / max(1, len(relevant))

    dcg = 0.0
    for rank, hit in enumerate(hits):
        if hit:
            dcg += 1.0 / math.log2(rank + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained cold10 checkpoint .pth")
    parser.add_argument("--cold_dataset", required=True, help="Example: amazon-beauty-cold10")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--out", default=None)
    parser.add_argument("--ks", default="10,20,50")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    print("Loading model checkpoint:", args.checkpoint)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.checkpoint
    )

    device = config["device"]
    model = model.to(device)
    model.eval()

    print("Loaded model:", config["model"])
    print("Dataset from checkpoint:", config["dataset"])
    print("Device:", device)
    print("use_connector:", config["use_connector"])
    print("emb_selection:", config["emb_selection"])

    split_dir = Path(args.dataset_dir) / args.cold_dataset
    removed_item_path = split_dir / "removed_items.item"
    removed_inter_path = split_dir / "removed_items.inter"
    cold_inter_path = split_dir / f"{args.cold_dataset}.inter"

    if not removed_item_path.exists():
        raise FileNotFoundError(removed_item_path)
    if not removed_inter_path.exists():
        raise FileNotFoundError(removed_inter_path)
    if not cold_inter_path.exists():
        raise FileNotFoundError(cold_inter_path)

    removed_items = pd.read_csv(removed_item_path, sep="\t")
    removed_inter = pd.read_csv(removed_inter_path, sep="\t")
    cold_inter = pd.read_csv(cold_inter_path, sep="\t")

    removed_item_col = find_col(removed_items, "item_id")
    removed_emb_col = find_col(removed_items, "llm_embedding")
    removed_inter_user_col = find_col(removed_inter, "user_id")
    removed_inter_item_col = find_col(removed_inter, "item_id")

    cold_inter_user_col = find_col(cold_inter, "user_id")
    cold_inter_item_col = find_col(cold_inter, "item_id")

    print("Removed item rows:", len(removed_items))
    print("Removed interaction rows:", len(removed_inter))

    # Raw token -> cold removed index
    removed_item_tokens = removed_items[removed_item_col].astype(str).tolist()
    removed_item_to_cold_idx = {tok: i for i, tok in enumerate(removed_item_tokens)}

    # Convert removed item LLM embeddings to tensor
    removed_emb = np.vstack([parse_float_seq(x) for x in removed_items[removed_emb_col]])
    removed_emb = torch.tensor(removed_emb, dtype=torch.float32, device=device)

    if not hasattr(model, "embedding_connector"):
        raise RuntimeError("Model has no embedding_connector. This script is for connector runs.")

    with torch.no_grad():
        # Trained user embeddings and seen item embeddings from cold10 model
        user_all, seen_item_all = model.forward()

        # Cold item embeddings inserted through connector
        cold_item_all = model.embedding_connector(removed_emb)

    # RecBole token mappings for users/items in the cold10 dataset
    user_map = build_token_map(dataset, config["USER_ID_FIELD"])
    seen_item_map = build_token_map(dataset, config["ITEM_ID_FIELD"])

    # Build training seen-item mask per user, so seen training positives are not recommended in seen+removed eval
    train_seen_by_user = {}
    for _, row in cold_inter.iterrows():
        u_tok = str(row[cold_inter_user_col])
        i_tok = str(row[cold_inter_item_col])
        if u_tok in user_map and i_tok in seen_item_map:
            u_inner = user_map[u_tok]
            i_inner = seen_item_map[i_tok]
            train_seen_by_user.setdefault(u_inner, set()).add(i_inner)

    # Build relevant removed items per user
    relevant_by_user = {}
    skipped_user = 0
    skipped_item = 0

    for _, row in removed_inter.iterrows():
        u_tok = str(row[removed_inter_user_col])
        i_tok = str(row[removed_inter_item_col])

        if u_tok not in user_map:
            skipped_user += 1
            continue
        if i_tok not in removed_item_to_cold_idx:
            skipped_item += 1
            continue

        u_inner = user_map[u_tok]
        cold_idx = removed_item_to_cold_idx[i_tok]
        relevant_by_user.setdefault(u_inner, set()).add(cold_idx)

    print("Users with removed-item ground truth:", len(relevant_by_user))
    print("Skipped removed interactions because user not in cold10 dataset:", skipped_user)
    print("Skipped removed interactions because item missing:", skipped_item)

    if len(relevant_by_user) == 0:
        raise RuntimeError("No evaluable users. The cold split may have removed users after filtering.")

    results = []

    with torch.no_grad():
        for u_inner, rel_cold_indices in relevant_by_user.items():
            u_vec = user_all[u_inner]

            # 1. Removed-only candidate set
            cold_scores = torch.matmul(cold_item_all, u_vec)
            removed_ranked = torch.argsort(cold_scores, descending=True).detach().cpu().numpy().tolist()

            row_result = {
                "user_inner_id": u_inner,
                "num_relevant_removed": len(rel_cold_indices),
            }

            for k in ks:
                r, n = recall_ndcg_from_ranks(removed_ranked, rel_cold_indices, k)
                row_result[f"removed_only_recall@{k}"] = r
                row_result[f"removed_only_ndcg@{k}"] = n

            # 2. Seen + removed candidate set
            seen_scores = torch.matmul(seen_item_all, u_vec)

            # mask padding / unknown item if index 0 exists
            if seen_scores.numel() > 0:
                seen_scores[0] = -1e20

            # mask user's known training items
            for seen_i in train_seen_by_user.get(u_inner, set()):
                if seen_i < seen_scores.numel():
                    seen_scores[seen_i] = -1e20

            all_scores = torch.cat([seen_scores, cold_scores], dim=0)
            all_ranked = torch.argsort(all_scores, descending=True).detach().cpu().numpy().tolist()

            offset = seen_item_all.shape[0]
            rel_all_indices = {offset + idx for idx in rel_cold_indices}

            for k in ks:
                r, n = recall_ndcg_from_ranks(all_ranked, rel_all_indices, k)
                row_result[f"seen_plus_removed_recall@{k}"] = r
                row_result[f"seen_plus_removed_ndcg@{k}"] = n

            results.append(row_result)

    result_df = pd.DataFrame(results)

    summary = {}
    for col in result_df.columns:
        if "recall@" in col or "ndcg@" in col:
            summary[col] = result_df[col].mean()

    print("\n===== PHASE 2 REMOVED-ITEM EVALUATION =====")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    if args.out is None:
        args.out = f"logs_connector/phase2_{args.cold_dataset}_{config['model']}_{config['emb_selection']}.csv"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print("\nPer-user results written to:")
    print(out_path)


if __name__ == "__main__":
    main()
