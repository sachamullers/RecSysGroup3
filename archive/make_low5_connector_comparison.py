from pathlib import Path
import pandas as pd


PHASE2_DIR = Path("logs_connector/phase2")

RUNS = [
    {
        "setting": "low5_plain_connector_norefilter",
        "method": "Low-5 plain connector",
        "summary_file": PHASE2_DIR / "beauty_low5_contgcn_var_noresid_run2_summary.csv",
    },
    {
        "setting": "low5_residual_connector_norefilter",
        "method": "Low-5 residual connector alpha=0.1",
        "summary_file": PHASE2_DIR / "beauty_low5_contgcn_var_resid_run2_summary.csv",
    },
]


def load_summary(run):
    path = run["summary_file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    df = pd.read_csv(path)

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "setting": run["setting"],
            "method": run["method"],
            "mode": row["mode"],
            "num_eval_users": row["num_eval_users"],
            "skipped_users": row["skipped_removed_interactions_user_not_in_cold_dataset"],
            "skipped_items": row["skipped_removed_interactions_item_missing"],

            "removed_only_recall@10": row["removed_only_recall@10"],
            "removed_only_ndcg@10": row["removed_only_ndcg@10"],
            "removed_only_recall@20": row["removed_only_recall@20"],
            "removed_only_ndcg@20": row["removed_only_ndcg@20"],
            "removed_only_recall@50": row["removed_only_recall@50"],
            "removed_only_ndcg@50": row["removed_only_ndcg@50"],

            "seen_plus_removed_recall@10": row["seen_plus_removed_recall@10"],
            "seen_plus_removed_ndcg@10": row["seen_plus_removed_ndcg@10"],
            "seen_plus_removed_recall@20": row["seen_plus_removed_recall@20"],
            "seen_plus_removed_ndcg@20": row["seen_plus_removed_ndcg@20"],
            "seen_plus_removed_recall@50": row["seen_plus_removed_recall@50"],
            "seen_plus_removed_ndcg@50": row["seen_plus_removed_ndcg@50"],

            "source_file": str(path),
        })

    return rows


def add_deltas(df):
    plain_connector = df[
        (df["setting"] == "low5_plain_connector_norefilter")
        & (df["mode"] == "connector")
    ]

    if plain_connector.empty:
        raise RuntimeError("Could not find plain connector baseline row.")

    base = plain_connector.iloc[0]

    for metric in [
        "removed_only_recall@10",
        "removed_only_ndcg@10",
        "seen_plus_removed_recall@10",
        "seen_plus_removed_ndcg@10",
    ]:
        df[f"delta_{metric}_vs_plain_connector"] = df[metric] - base[metric]

    return df


def main():
    rows = []
    for run in RUNS:
        rows.extend(load_summary(run))

    df = pd.DataFrame(rows)

    # Sanity check: both runs should be comparable
    users_by_setting = df.groupby("setting")["num_eval_users"].nunique()
    if not all(users_by_setting == 1):
        raise RuntimeError("Each setting should have exactly one num_eval_users value.")

    unique_user_counts = df.groupby("setting")["num_eval_users"].first()
    if len(set(unique_user_counts.tolist())) != 1:
        print("WARNING: settings have different num_eval_users:")
        print(unique_user_counts)

    skipped = df.groupby("setting")[["skipped_users", "skipped_items"]].max()
    if (skipped > 0).any().any():
        print("WARNING: some runs skipped users/items:")
        print(skipped)

    df = add_deltas(df)

    mode_order = {"random": 0, "direct": 1, "connector": 2}
    setting_order = {
        "low5_plain_connector_norefilter": 0,
        "low5_residual_connector_norefilter": 1,
    }

    df["setting_order"] = df["setting"].map(setting_order)
    df["mode_order"] = df["mode"].map(mode_order)
    df = df.sort_values(["setting_order", "mode_order"])
    df = df.drop(columns=["setting_order", "mode_order"])

    out_path = PHASE2_DIR / "beauty_low5_connector_comparison_run2.csv"
    df.to_csv(out_path, index=False)

    print("\nWrote:")
    print(out_path)

    print("\nReadable comparison:")
    view_cols = [
        "method",
        "mode",
        "num_eval_users",
        "removed_only_recall@10",
        "removed_only_ndcg@10",
        "seen_plus_removed_recall@10",
        "seen_plus_removed_ndcg@10",
    ]
    view = df[view_cols].copy()
    view = view.rename(columns={
        "method": "Run",
        "mode": "Mode",
        "num_eval_users": "Users",
        "removed_only_recall@10": "Cold-only R@10",
        "removed_only_ndcg@10": "Cold-only N@10",
        "seen_plus_removed_recall@10": "Cold-vs-seen R@10",
        "seen_plus_removed_ndcg@10": "Cold-vs-seen N@10",
    })

    for col in view.columns:
        if col not in ["Run", "Mode", "Users"]:
            view[col] = view[col].map(lambda x: f"{x:.6f}")

    print(view.to_string(index=False))

    print("\nMain connector comparison only:")
    main = df[df["mode"] == "connector"].copy()
    main_cols = [
        "method",
        "removed_only_recall@10",
        "removed_only_ndcg@10",
        "seen_plus_removed_recall@10",
        "seen_plus_removed_ndcg@10",
        "delta_removed_only_ndcg@10_vs_plain_connector",
        "delta_seen_plus_removed_ndcg@10_vs_plain_connector",
    ]
    main_view = main[main_cols].copy()
    main_view = main_view.rename(columns={
        "method": "Method",
        "removed_only_recall@10": "Cold-only R@10",
        "removed_only_ndcg@10": "Cold-only N@10",
        "seen_plus_removed_recall@10": "Cold-vs-seen R@10",
        "seen_plus_removed_ndcg@10": "Cold-vs-seen N@10",
        "delta_removed_only_ndcg@10_vs_plain_connector": "Delta Cold-only N@10 vs plain",
        "delta_seen_plus_removed_ndcg@10_vs_plain_connector": "Delta Cold-vs-seen N@10 vs plain",
    })

    for col in main_view.columns:
        if col != "Method":
            main_view[col] = main_view[col].map(lambda x: f"{x:.6f}")

    print(main_view.to_string(index=False))


if __name__ == "__main__":
    main()