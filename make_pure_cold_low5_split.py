import argparse
import os
import numpy as np
import pandas as pd


def get_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"Could not find column starting with {prefix}. Columns: {list(df.columns)}")
    return matches[0]


def iterative_user_kcore(inter, user_col, min_user=5):
    prev_len = -1
    cur = inter.copy()

    while prev_len != len(cur):
        prev_len = len(cur)
        user_counts = cur[user_col].value_counts()
        keep_users = set(user_counts[user_counts >= min_user].index)
        cur = cur[cur[user_col].isin(keep_users)]

    return cur.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--suffix", default="low5cold")
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--warm_item_min_inter", type=int, default=5)
    parser.add_argument("--min_user", type=int, default=5)
    args = parser.parse_args()

    src_dir = f"dataset/{args.dataset}"
    out_dataset = f"{args.dataset}-{args.suffix}"
    out_dir = f"dataset/{out_dataset}"
    os.makedirs(out_dir, exist_ok=True)

    inter = pd.read_csv(f"{src_dir}/{args.dataset}.inter", sep="\t")
    item = pd.read_csv(f"{src_dir}/{args.dataset}.item", sep="\t")

    user_col = get_col(inter, "user_id")
    item_col = get_col(inter, "item_id")
    rating_cols = [c for c in inter.columns if c.startswith("rating")]
    item_item_col = get_col(item, "item_id")

    print("Original interactions:", len(inter))
    print("Original item rows:", len(item))

    if rating_cols:
        rating_col = rating_cols[0]
        inter[rating_col] = pd.to_numeric(inter[rating_col], errors="coerce")
        inter = inter[inter[rating_col] >= args.min_rating].copy()
        print(f"After rating >= {args.min_rating}:", len(inter))

    item_counts = inter[item_col].value_counts()

    warm_items = set(item_counts[item_counts >= args.warm_item_min_inter].index)
    cold_items = set(item_counts[item_counts < args.warm_item_min_inter].index)

    print("Warm active items before user filtering:", len(warm_items))
    print("Cold active items before user filtering:", len(cold_items))

    train_inter_raw = inter[inter[item_col].isin(warm_items)].copy()
    removed_inter_raw = inter[inter[item_col].isin(cold_items)].copy()

    train_inter = iterative_user_kcore(
        train_inter_raw,
        user_col=user_col,
        min_user=args.min_user,
    )

    final_train_users = set(train_inter[user_col].unique())
    final_train_items = set(train_inter[item_col].unique())

    removed_inter = removed_inter_raw[removed_inter_raw[user_col].isin(final_train_users)].copy()
    final_removed_items = set(removed_inter[item_col].unique())

    train_item = item[item[item_item_col].isin(final_train_items)].copy()
    removed_item = item[item[item_item_col].isin(final_removed_items)].copy()

    overlap = final_train_items.intersection(final_removed_items)

    train_inter.to_csv(f"{out_dir}/{out_dataset}.inter", sep="\t", index=False)
    train_item.to_csv(f"{out_dir}/{out_dataset}.item", sep="\t", index=False)
    removed_inter.to_csv(f"{out_dir}/removed_items.inter", sep="\t", index=False)
    removed_item.to_csv(f"{out_dir}/removed_items.item", sep="\t", index=False)

    print()
    print("Pure cold-start dataset:", out_dataset)
    print("Final train users:", len(final_train_users))
    print("Final train warm items:", len(final_train_items))
    print("Final train interactions:", len(train_inter))
    print("Final cold eval items:", len(final_removed_items))
    print("Final cold eval interactions:", len(removed_inter))
    print("Overlap between train and cold items:", len(overlap))

    if len(overlap) != 0:
        raise RuntimeError("Train/cold item overlap is not zero.")
    if len(removed_inter) == 0:
        raise RuntimeError("No cold interactions left for evaluation.")


if __name__ == "__main__":
    main()
