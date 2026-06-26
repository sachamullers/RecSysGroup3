import argparse
import os
import numpy as np
import pandas as pd
import torch

# RecBole / old numpy compatibility
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

from recbole.quick_start import load_data_and_model


def get_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"Could not find column starting with {prefix}. Columns: {list(df.columns)}")
    return matches[0]


def parse_embedding_column(series):
    return np.stack(
        series.astype(str)
        .apply(lambda s: np.array([float(x) for x in s.split()], dtype=np.float32))
        .values
    )


def id_to_token(dataset, field, internal_id):
    try:
        return str(dataset.id2token(field, [int(internal_id)])[0])
    except Exception:
        try:
            return str(dataset.field2id_token[field][int(internal_id)])
        except Exception:
            return str(internal_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cold_dataset", required=True)
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--modes", default="random,direct,connector")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.checkpoint
    )

    device = config["device"]
    model = model.to(device)
    model.eval()

    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    user_token_to_internal = dataset.field2token_id[uid_field]
    item_token_to_internal = dataset.field2token_id[iid_field]

    cold_dir = os.path.join(args.dataset_dir, args.cold_dataset)

    removed_item_path = os.path.join(cold_dir, "removed_items.item")
    removed_inter_path = os.path.join(cold_dir, "removed_items.inter")
    train_inter_path = os.path.join(cold_dir, f"{args.cold_dataset}.inter")

    removed_item = pd.read_csv(removed_item_path, sep="\t")
    removed_inter = pd.read_csv(removed_inter_path, sep="\t")
    train_inter = pd.read_csv(train_inter_path, sep="\t")

    removed_item_col = get_col(removed_item, "item_id")
    removed_emb_col = get_col(removed_item, "llm_embedding")

    removed_inter_user_col = get_col(removed_inter, "user_id")
    removed_inter_item_col = get_col(removed_inter, "item_id")

    train_user_col = get_col(train_inter, "user_id")
    train_item_col = get_col(train_inter, "item_id")

    removed_tokens = removed_item[removed_item_col].astype(str).tolist()
    removed_emb_np = parse_embedding_column(removed_item[removed_emb_col])
    removed_emb = torch.tensor(removed_emb_np, dtype=torch.float32, device=device)

    # Relevant cold items per user
    relevant_cold_by_user = {}
    for _, row in removed_inter.iterrows():
        u = str(row[removed_inter_user_col])
        i = str(row[removed_inter_item_col])
        if u in user_token_to_internal:
            relevant_cold_by_user.setdefault(u, set()).add(i)

    # Seen train positives per user, for masking in mixed ranking
    train_seen_by_user_internal = {}
    for _, row in train_inter.iterrows():
        u = str(row[train_user_col])
        i = str(row[train_item_col])
        if u in user_token_to_internal and i in item_token_to_internal:
            u_internal = int(user_token_to_internal[u])
            i_internal = int(item_token_to_internal[i])
            train_seen_by_user_internal.setdefault(u_internal, set()).add(i_internal)

    eval_user_tokens = [
        u for u in relevant_cold_by_user.keys()
        if u in user_token_to_internal
    ]
    eval_user_tokens = eval_user_tokens[:args.num_users]

    print("Loaded model:", model.__class__.__name__)
    print("Dataset:", args.cold_dataset)
    print("Modes:", modes)
    print("Top-k:", args.topk)
    print("Users exported:", len(eval_user_tokens))
    print("Removed items:", len(removed_tokens))

    with torch.no_grad():
        user_all, seen_item_all = model.forward()

        sampled_indices = model.get_sampled_indices().to(device)

        cold_vectors = {}

        if "random" in modes:
            torch.manual_seed(int(config["seed"]))
            cold_vectors["random"] = torch.randn(
                removed_emb.shape[0],
                int(config["embedding_size"]),
                device=device,
            )

        if "direct" in modes:
            cold_vectors["direct"] = removed_emb[:, sampled_indices]

        if "connector" in modes:
            if hasattr(model, "embedding_connector"):
                cold_vectors["connector"] = model.embedding_connector(removed_emb)
            elif hasattr(model, "item_embedding_connector"):
                cold_vectors["connector"] = model.item_embedding_connector(removed_emb)
            else:
                raise AttributeError("Could not find embedding_connector or item_embedding_connector.")

        rows = []

        for user_token in eval_user_tokens:
            u_internal = int(user_token_to_internal[user_token])
            user_vec = user_all[u_internal]

            relevant_cold = relevant_cold_by_user.get(user_token, set())
            train_seen_items = train_seen_by_user_internal.get(u_internal, set())

            for mode, cold_item_all in cold_vectors.items():
                # ------------------------------------------------------------
                # Setting 1: removed_only
                # ------------------------------------------------------------
                cold_scores = torch.matmul(cold_item_all, user_vec)

                k_removed = min(args.topk, cold_scores.shape[0])
                top_scores, top_indices = torch.topk(cold_scores, k=k_removed)

                for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
                    item_token = removed_tokens[int(idx)]
                    rows.append({
                        "user_id": user_token,
                        "setting": "removed_only",
                        "mode": mode,
                        "rank": rank,
                        "item_id": item_token,
                        "candidate_type": "cold_removed",
                        "score": score,
                        "is_relevant_cold_item": int(item_token in relevant_cold),
                    })

                # ------------------------------------------------------------
                # Setting 2: seen_plus_removed
                # seen candidates = model's seen items from training dataset
                # cold candidates = removed items embedded by the mode
                # ------------------------------------------------------------
                seen_scores = torch.matmul(seen_item_all, user_vec)

                # Mask padding item id 0 if present
                if seen_scores.shape[0] > 0:
                    seen_scores[0] = -1e20

                # Mask user's training positives so we do not recommend items
                # the user already interacted with in the warm training set.
                for pos_i in train_seen_items:
                    if 0 <= pos_i < seen_scores.shape[0]:
                        seen_scores[pos_i] = -1e20

                mixed_scores = torch.cat([seen_scores, cold_scores], dim=0)

                k_mixed = min(args.topk, mixed_scores.shape[0])
                mixed_top_scores, mixed_top_indices = torch.topk(mixed_scores, k=k_mixed)

                seen_count = seen_scores.shape[0]

                for rank, (score, idx) in enumerate(zip(mixed_top_scores.tolist(), mixed_top_indices.tolist()), start=1):
                    idx = int(idx)

                    if idx < seen_count:
                        item_token = id_to_token(dataset, iid_field, idx)
                        candidate_type = "seen_train"
                        is_relevant = 0
                    else:
                        cold_idx = idx - seen_count
                        item_token = removed_tokens[cold_idx]
                        candidate_type = "cold_removed"
                        is_relevant = int(item_token in relevant_cold)

                    rows.append({
                        "user_id": user_token,
                        "setting": "seen_plus_removed",
                        "mode": mode,
                        "rank": rank,
                        "item_id": item_token,
                        "candidate_type": candidate_type,
                        "score": score,
                        "is_relevant_cold_item": is_relevant,
                    })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print("Wrote:", args.out)
    print("Rows:", len(out_df))


if __name__ == "__main__":
    main()
