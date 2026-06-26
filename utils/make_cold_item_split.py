import argparse
import os
import numpy as np
import pandas as pd


def get_col(df, prefix):
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(f"Could not find column starting with {prefix}. Columns: {list(df.columns)}")
    return matches[0]


def iterative_kcore(inter, user_col, item_col, min_user=5, min_item=5):
    """
    Repeatedly filter until all users/items satisfy min interactions.
    This approximates RecBole's user_inter_num_interval/item_inter_num_interval filtering.
    """
    prev_len = -1
    cur = inter.copy()

    while prev_len != len(cur):
        prev_len = len(cur)

        if min_user is not None and min_user > 0:
            user_counts = cur[user_col].value_counts()
            keep_users = set(user_counts[user_counts >= min_user].index)
            cur = cur[cur[user_col].isin(keep_users)]

        if min_item is not None and min_item > 0:
            item_counts = cur[item_col].value_counts()
            keep_items = set(item_counts[item_counts >= min_item].index)
            cur = cur[cur[item_col].isin(keep_items)]

    return cur.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--remove_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--min_user", type=int, default=5)
    parser.add_argument("--min_item", type=int, default=5)
    parser.add_argument("--suffix", default="cold10")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    src_dir = f"dataset/{args.dataset}"
    out_dataset = f"{args.dataset}-{args.suffix}"
    out_dir = f"dataset/{out_dataset}"
    os.makedirs(out_dir, exist_ok=True)

    inter_path = f"{src_dir}/{args.dataset}.inter"
    item_path = f"{src_dir}/{args.dataset}.item"

    inter = pd.read_csv(inter_path, sep="\t")
    item = pd.read_csv(item_path, sep="\t")

    user_col = get_col(inter, "user_id")
    item_col = get_col(inter, "item_id")
    rating_cols = [c for c in inter.columns if c.startswith("rating")]

    item_item_col = get_col(item, "item_id")

    print("Original interaction rows:", len(inter))
    print("Original item rows:", len(item))

    # 1. Apply rating filter first, matching YAML val_interval rating >= 4
    if rating_cols and args.min_rating is not None:
        rating_col = rating_cols[0]
        inter[rating_col] = pd.to_numeric(inter[rating_col], errors="coerce")
        inter = inter[inter[rating_col] >= args.min_rating].copy()
        print(f"After rating >= {args.min_rating}:", len(inter))

    # 2. Apply original 5-core filtering before choosing cold items
    core_inter = iterative_kcore(
        inter,
        user_col=user_col,
        item_col=item_col,
        min_user=args.min_user,
        min_item=args.min_item,
    )

    core_users = set(core_inter[user_col].unique())
    core_items = np.array(sorted(core_inter[item_col].unique()))

    print("After initial k-core:")
    print("  users:", len(core_users))
    print("  items:", len(core_items))
    print("  interactions:", len(core_inter))

    # 3. Remove 10% of active core items, not all metadata items
    n_remove = max(1, int(len(core_items) * args.remove_ratio))
    removed_items_initial = set(rng.choice(core_items, size=n_remove, replace=False))

    train_inter_raw = core_inter[~core_inter[item_col].isin(removed_items_initial)].copy()
    removed_inter_raw = core_inter[core_inter[item_col].isin(removed_items_initial)].copy()

    # 4. Apply k-core again to train split, because removing items can drop users/items below 5
    train_inter = iterative_kcore(
        train_inter_raw,
        user_col=user_col,
        item_col=item_col,
        min_user=args.min_user,
        min_item=args.min_item,
    )

    final_train_users = set(train_inter[user_col].unique())
    final_train_items = set(train_inter[item_col].unique())

    # 5. Keep removed-item eval interactions only for users still present in training
    removed_inter = removed_inter_raw[removed_inter_raw[user_col].isin(final_train_users)].copy()
    final_removed_items = set(removed_inter[item_col].unique())

    # Keep only item features actually used in train/eval
    train_item = item[item[item_item_col].isin(final_train_items)].copy()
    removed_item = item[item[item_item_col].isin(final_removed_items)].copy()

    # Safety checks
    overlap = final_train_items.intersection(final_removed_items)

    train_inter.to_csv(f"{out_dir}/{out_dataset}.inter", sep="\t", index=False)
    train_item.to_csv(f"{out_dir}/{out_dataset}.item", sep="\t", index=False)

    removed_inter.to_csv(f"{out_dir}/removed_items.inter", sep="\t", index=False)
    removed_item.to_csv(f"{out_dir}/removed_items.item", sep="\t", index=False)

    print()
    print("Original dataset:", args.dataset)
    print("Cold-start training dataset:", out_dataset)
    print()
    print("Initial active core items:", len(core_items))
    print("Initially selected removed items:", len(removed_items_initial))
    print()
    print("Final train users:", len(final_train_users))
    print("Final train items:", len(final_train_items))
    print("Final train interactions:", len(train_inter))
    print()
    print("Final removed eval items:", len(final_removed_items))
    print("Final removed eval interactions:", len(removed_inter))
    print("Overlap between train and removed items:", len(overlap))
    print()
    print("Training files:")
    print(f"{out_dir}/{out_dataset}.inter")
    print(f"{out_dir}/{out_dataset}.item")
    print()
    print("Cold-item evaluation files:")
    print(f"{out_dir}/removed_items.inter")
    print(f"{out_dir}/removed_items.item")

    if len(overlap) != 0:
        raise RuntimeError("Train/removed item overlap is not zero. Something is wrong.")

    if len(removed_inter) == 0:
        raise RuntimeError("No removed interactions left for evaluation. Try lower remove_ratio.")


if __name__ == "__main__":
    main()