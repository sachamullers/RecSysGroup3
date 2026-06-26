"""
Script to conduct evaluation of cold-start item recommendation using a trained RecBole model checkpoint.

    random:
        removed item = random 128-dim vector

    direct:
        removed item = direct LLMInit rand/uni/var selected dimensions

    connector:
        removed item = connector(768-dim removed item LLM embedding)

It reports:
    removed_only:
        Rank correct removed items only among removed items.
        Easier sanity check.

    seen_plus_removed:
        Rank correct removed items against seen items + removed items.
        Harder and more realistic.
"""

import argparse
import math
from pathlib import Path

import numpy as np

# Compatibility patch for older RecBole with newer NumPy
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

import pandas as pd
import torch
from recbole.quick_start.quick_start import load_data_and_model



def find_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"Could not find column starting with '{prefix}'. Columns: {list(df.columns)}")
    return matches[0]


def parse_float_seq(x):
    return np.array([float(v) for v in str(x).split()], dtype=np.float32)


def build_token_map(dataset, field):
    """
    Converts RecBole's token mapping to raw_token -> inner_id.
    """
    token_info = dataset.field2token_id[field]

    if isinstance(token_info, dict):
        return {str(k): int(v) for k, v in token_info.items()}

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


def get_sampled_indices_from_model(model, emb_selection):
    """
    Uses the same rand/uni/var dimension selection logic as the trained model.
    This is needed for the direct LLMInit baseline.
    """
    if hasattr(model, "get_sampled_indices"):
        return model.get_sampled_indices()

    if emb_selection == "rand":
        return model.random_sample()
    if emb_selection == "uni":
        return model.even_sample()
    if emb_selection == "var":
        return model.var_sample()

    raise RuntimeError(f"Cannot determine sampled indices for emb_selection={emb_selection}")


def make_cold_item_embeddings(mode, model, removed_emb, config, device):
    """
    Creates embeddings for removed/new items using one of three methods:

    random:
        random 128-dim vectors

    direct:
        direct LLMInit projection, i.e. selected rand/uni/var dimensions

    connector:
        connector(768-dim removed item LLM embedding)
    """
    embedding_size = int(config["embedding_size"])
    emb_selection = config["emb_selection"]

    if mode == "connector":
        if not hasattr(model, "embedding_connector"):
            raise RuntimeError("Model has no embedding_connector. Cannot use connector mode.")
        return model.embedding_connector(removed_emb)

    if mode == "direct":
        sampled_indices = get_sampled_indices_from_model(model, emb_selection).to(device)
        return removed_emb[:, sampled_indices]

    if mode == "random":
        torch.manual_seed(int(config["seed"]))
        return torch.randn(
            removed_emb.shape[0],
            embedding_size,
            device=device,
        )

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_one_mode(
    mode,
    model,
    config,
    dataset,
    user_all,
    seen_item_all,
    removed_emb,
    removed_items,
    removed_inter,
    cold_inter,
    ks,
    device,
):
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]

    user_map = build_token_map(dataset, user_field)
    seen_item_map = build_token_map(dataset, item_field)

    removed_item_col = find_col(removed_items, "item_id")
    removed_inter_user_col = find_col(removed_inter, "user_id")
    removed_inter_item_col = find_col(removed_inter, "item_id")

    cold_inter_user_col = find_col(cold_inter, "user_id")
    cold_inter_item_col = find_col(cold_inter, "item_id")

    removed_item_tokens = removed_items[removed_item_col].astype(str).tolist()
    removed_item_to_cold_idx = {tok: i for i, tok in enumerate(removed_item_tokens)}

    with torch.no_grad():
        cold_item_all = make_cold_item_embeddings(
            mode=mode,
            model=model,
            removed_emb=removed_emb,
            config=config,
            device=device,
        )

    train_seen_by_user = {}
    for _, row in cold_inter.iterrows():
        u_tok = str(row[cold_inter_user_col])
        i_tok = str(row[cold_inter_item_col])

        if u_tok in user_map and i_tok in seen_item_map:
            u_inner = user_map[u_tok]
            i_inner = seen_item_map[i_tok]
            train_seen_by_user.setdefault(u_inner, set()).add(i_inner)

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

    if len(relevant_by_user) == 0:
        raise RuntimeError(
            "No evaluable users found. This means users in removed_items.inter "
            "are not present in the cold10 RecBole dataset after filtering."
        )

    rows = []

    with torch.no_grad():
        for u_inner, rel_cold_indices in relevant_by_user.items():
            u_vec = user_all[u_inner]

            cold_scores = torch.matmul(cold_item_all, u_vec)

            # Evaluation A: rank removed items only
            removed_ranked = torch.argsort(cold_scores, descending=True).detach().cpu().numpy().tolist()

            row_result = {
                "mode": mode,
                "user_inner_id": int(u_inner),
                "num_relevant_removed": len(rel_cold_indices),
            }

            for k in ks:
                r, n = recall_ndcg_from_ranks(removed_ranked, rel_cold_indices, k)
                row_result[f"removed_only_recall@{k}"] = r
                row_result[f"removed_only_ndcg@{k}"] = n

            # Evaluation B: rank seen items + removed items
            seen_scores = torch.matmul(seen_item_all, u_vec).clone()

            # Mask padding/unknown item if present
            if seen_scores.numel() > 0:
                seen_scores[0] = -1e20

            # Mask user's known training positives
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

            rows.append(row_result)

    result_df = pd.DataFrame(rows)

    summary = {
        "mode": mode,
        "num_eval_users": len(relevant_by_user),
        "skipped_removed_interactions_user_not_in_cold_dataset": skipped_user,
        "skipped_removed_interactions_item_missing": skipped_item,
    }

    for col in result_df.columns:
        if "recall@" in col or "ndcg@" in col:
            summary[col] = result_df[col].mean()

    return result_df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained cold10 checkpoint .pth")
    parser.add_argument("--cold_dataset", required=True, help="Example: amazon-beauty-cold10")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument(
        "--modes",
        default="random,direct,connector",
        help="Comma-separated list from: random,direct,connector",
    )
    parser.add_argument("--ks", default="10,20,50")
    parser.add_argument("--out_dir", default="logs_connector/phase2")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    allowed_modes = {"random", "direct", "connector"}
    bad_modes = [m for m in modes if m not in allowed_modes]
    if bad_modes:
        raise ValueError(f"Invalid modes: {bad_modes}. Allowed: {sorted(allowed_modes)}")

    print("Loading checkpoint:", args.checkpoint)
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.checkpoint
    )

    device = config["device"]
    model = model.to(device)
    model.eval()

    print("Loaded model:", config["model"])
    print("Checkpoint dataset:", config["dataset"])
    print("Requested cold dataset:", args.cold_dataset)
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

    removed_emb_col = find_col(removed_items, "llm_embedding")
    removed_emb = np.vstack([parse_float_seq(x) for x in removed_items[removed_emb_col]])
    removed_emb = torch.tensor(removed_emb, dtype=torch.float32, device=device)

    print("Removed items:", len(removed_items))
    print("Removed interactions:", len(removed_inter))
    print("Removed embedding shape:", tuple(removed_emb.shape))

    with torch.no_grad():
        user_all, seen_item_all = model.forward()

    print("Trained user embedding shape:", tuple(user_all.shape))
    print("Seen item embedding shape:", tuple(seen_item_all.shape))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prefix is None:
        prefix = f"{args.cold_dataset}_{config['model']}_{config['emb_selection']}"
    else:
        prefix = args.prefix

    all_summaries = []
    all_user_results = []

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"Evaluating mode: {mode}")
        print("=" * 80)

        result_df, summary = evaluate_one_mode(
            mode=mode,
            model=model,
            config=config,
            dataset=dataset,
            user_all=user_all,
            seen_item_all=seen_item_all,
            removed_emb=removed_emb,
            removed_items=removed_items,
            removed_inter=removed_inter,
            cold_inter=cold_inter,
            ks=ks,
            device=device,
        )

        all_summaries.append(summary)
        all_user_results.append(result_df)

        user_out = out_dir / f"{prefix}_{mode}_per_user.csv"
        result_df.to_csv(user_out, index=False)

        print(f"\nPer-user results written to: {user_out}")
        print("\nSummary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    summary_df = pd.DataFrame(all_summaries)
    summary_out = out_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    combined_user_df = pd.concat(all_user_results, ignore_index=True)
    combined_user_out = out_dir / f"{prefix}_all_modes_per_user.csv"
    combined_user_df.to_csv(combined_user_out, index=False)

    print("\n" + "=" * 80)
    print("FINAL PHASE 2 SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\nSummary written to:")
    print(summary_out)
    print("\nCombined per-user results written to:")
    print(combined_user_out)


if __name__ == "__main__":
    main()
