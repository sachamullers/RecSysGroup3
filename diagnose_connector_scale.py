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


def summarize_vector_norms(name, x):
    norms = torch.norm(x, dim=1).detach().cpu().numpy()
    return {
        "name": name,
        "count": len(norms),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_median": float(np.median(norms)),
        "norm_max": float(np.max(norms)),
    }


def summarize_scores(name, scores):
    scores = scores.detach().cpu().numpy().reshape(-1)
    return {
        "name": name,
        "score_count": len(scores),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cold_dataset", required=True)
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--out", default=None)
    parser.add_argument("--num_users", type=int, default=512)
    parser.add_argument("--num_items", type=int, default=512)
    args = parser.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.checkpoint
    )

    device = config["device"]
    model = model.to(device)
    model.eval()

    if args.out is None:
        args.out = f"logs_connector/phase2/{args.cold_dataset}_diagnostic_scale.csv"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    removed_item_path = os.path.join(
        args.dataset_dir,
        args.cold_dataset,
        "removed_items.item",
    )

    removed_item = pd.read_csv(removed_item_path, sep="\t")
    removed_item_col = get_col(removed_item, "item_id")
    removed_emb_col = get_col(removed_item, "llm_embedding")

    removed_tokens = removed_item[removed_item_col].astype(str).tolist()

    removed_emb_np = np.stack(
        removed_item[removed_emb_col]
        .astype(str)
        .apply(lambda s: np.array([float(x) for x in s.split()], dtype=np.float32))
        .values
    )

    removed_emb = torch.tensor(removed_emb_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        # Final propagated embeddings from the trained model
        user_all, seen_item_all = model.forward()

        # Direct cold baseline: same var/rand/uni selected dimensions
        sampled_indices = model.get_sampled_indices().to(device)
        direct_cold = removed_emb[:, sampled_indices]

        # Connector cold embeddings
        if hasattr(model, "item_embedding_connector"):
            connector_cold = model.item_embedding_connector(removed_emb)
        elif hasattr(model, "embedding_connector"):
            connector_cold = model.embedding_connector(removed_emb)
        else:
            raise AttributeError("Could not find connector module on model.")

        # sample users/items so score matrix is not huge
        n_users = min(args.num_users, user_all.shape[0])
        n_seen = min(args.num_items, seen_item_all.shape[0])
        n_cold = min(args.num_items, direct_cold.shape[0])

        user_sample = user_all[:n_users]
        seen_sample = seen_item_all[:n_seen]
        direct_sample = direct_cold[:n_cold]
        connector_sample = connector_cold[:n_cold]

        seen_scores = torch.matmul(user_sample, seen_sample.T)
        direct_scores = torch.matmul(user_sample, direct_sample.T)
        connector_scores = torch.matmul(user_sample, connector_sample.T)

    rows = []
    rows.append(summarize_vector_norms("user_final", user_all))
    rows.append(summarize_vector_norms("seen_item_final", seen_item_all))
    rows.append(summarize_vector_norms("cold_direct_var", direct_cold))
    rows.append(summarize_vector_norms("cold_connector", connector_cold))

    score_rows = []
    score_rows.append(summarize_scores("user_dot_seen_item", seen_scores))
    score_rows.append(summarize_scores("user_dot_cold_direct", direct_scores))
    score_rows.append(summarize_scores("user_dot_cold_connector", connector_scores))

    norm_df = pd.DataFrame(rows)
    score_df = pd.DataFrame(score_rows)

    print("\nEmbedding norm summary:")
    print(norm_df.to_string(index=False))

    print("\nScore summary:")
    print(score_df.to_string(index=False))

    # Save combined CSV with separate sections encoded by type
    norm_df.insert(0, "section", "embedding_norms")
    score_df.insert(0, "section", "dot_product_scores")

    out_df = pd.concat([norm_df, score_df], ignore_index=True, sort=False)
    out_df.to_csv(args.out, index=False)

    print("\nWrote:", args.out)


if __name__ == "__main__":
    main()
