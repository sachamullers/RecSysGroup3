#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path

import pandas as pd
import torch

_original_torch_load = torch.load

def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    kwargs.setdefault("map_location", "cpu")
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_compat

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk


def get_eval_users(test_data, dataset):
    # Prefer RecBole's eval-user list if available.
    for attr in ["uid_list", "uid2history_item"]:
        if hasattr(test_data, attr):
            value = getattr(test_data, attr)
            if attr == "uid_list":
                try:
                    return list(value.cpu().numpy())
                except Exception:
                    return list(value)
            if attr == "uid2history_item":
                return list(range(1, len(value)))

    # Fallback: all users except padding id 0.
    return list(range(1, dataset.user_num))


def to_external_ids(dataset, field, ids):
    try:
        return dataset.id2token(field, ids)
    except Exception:
        return ids


def export_topk(model_file, output, k, batch_size, external_ids):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=str(model_file)
    )

    device = config["device"]
    user_field = config["USER_ID_FIELD"]
    item_field = config["ITEM_ID_FIELD"]

    users = get_eval_users(test_data, dataset)
    rows = []

    print(f"Loaded checkpoint: {model_file}")
    print(f"Dataset: {config['dataset']}")
    print(f"Model: {config['model']}")
    print(f"Users to export: {len(users):,}")
    print(f"Top-k: {k}")

    # Avoid CPU/CUDA mismatch in RecBole cached full-sort embeddings.
    if hasattr(model, "restore_user_e"):
        model.restore_user_e = None
    if hasattr(model, "restore_item_e"):
        model.restore_item_e = None

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]

        topk_scores, topk_items = full_sort_topk(
            batch_users,
            model,
            test_data,
            k=k,
            device=device,
        )

        topk_scores = topk_scores.detach().cpu().numpy()
        topk_items = topk_items.detach().cpu().numpy()

        if external_ids:
            out_users = to_external_ids(dataset, user_field, batch_users)
            flat_items = topk_items.reshape(-1)
            out_items = to_external_ids(dataset, item_field, flat_items).reshape(topk_items.shape)
        else:
            out_users = batch_users
            out_items = topk_items

        for row_idx, user_id in enumerate(out_users):
            for rank_idx in range(k):
                rows.append(
                    {
                        "user_id": user_id,
                        "item_id": out_items[row_idx, rank_idx],
                        "rank": rank_idx + 1,
                        "score": float(topk_scores[row_idx, rank_idx]),
                    }
                )

        print(f"Exported users {min(start + batch_size, len(users)):,}/{len(users):,}")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--external-ids", action="store_true")
    args = parser.parse_args()

    export_topk(
        model_file=args.model_file,
        output=args.output,
        k=args.k,
        batch_size=args.batch_size,
        external_ids=args.external_ids,
    )


if __name__ == "__main__":
    main()
