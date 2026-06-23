#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _safe_import_matplotlib():
    import matplotlib.pyplot as plt
    return plt


def read_table(path: Path, sep: str) -> pd.DataFrame:
    if sep == "\\t":
        sep = "\t"
    return pd.read_csv(path, sep=sep)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "user_id:token": "user_id",
            "item_id:token": "item_id",
            "brand:token": "brand",
            "rating:float": "rating",
            "timestamp:float": "timestamp",
        }
    )


def restrict_top_k(recommendations: pd.DataFrame, k: int) -> pd.DataFrame:
    recs = recommendations.copy()

    if "rank" not in recs.columns:
        recs["rank"] = recs.groupby("user_id").cumcount() + 1

    recs["rank"] = pd.to_numeric(recs["rank"], errors="coerce")
    recs = recs.dropna(subset=["rank"])
    recs["rank"] = recs["rank"].astype(int)

    return recs.sort_values(["user_id", "rank"]).query("rank <= @k")


def catalog_coverage_at_k(recommendations_k: pd.DataFrame, items: pd.DataFrame) -> float:
    total_items = items["item_id"].nunique()
    if total_items == 0:
        return 0.0
    return recommendations_k["item_id"].nunique() / total_items


def brand_diversity_per_user_at_k(
    recommendations_k: pd.DataFrame,
    items: pd.DataFrame,
) -> pd.Series:
    recs = recommendations_k.merge(
        items[["item_id", "brand"]],
        on="item_id",
        how="left",
    )

    if recs.empty:
        return pd.Series(dtype=float, name="brand_diversity")

    per_user = recs.groupby("user_id").agg(
        unique_brands=("brand", lambda x: x.dropna().nunique()),
        recommended_items=("item_id", "count"),
    )

    return (per_user["unique_brands"] / per_user["recommended_items"]).rename(
        "brand_diversity"
    )


def brand_loyal_groups(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    history = interactions[["user_id", "item_id"]].merge(
        items[["item_id", "brand"]],
        on="item_id",
        how="left",
    ).dropna(subset=["brand"])

    if history.empty:
        return pd.Series(dtype=bool, name="brand_loyal")

    counts = (
        history.groupby(["user_id", "brand"])
        .size()
        .rename("brand_count")
        .reset_index()
    )

    totals = history.groupby("user_id").size().rename("total_count")
    counts = counts.merge(totals, on="user_id")
    counts["brand_share"] = counts["brand_count"] / counts["total_count"]

    max_share = counts.groupby("user_id")["brand_share"].max()
    return (max_share >= threshold).rename("brand_loyal")


def item_tail_bucket_scores(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    num_buckets: int,
) -> pd.DataFrame:
    popularity = interactions["item_id"].value_counts().rename("interaction_count")

    bucket_df = items[["item_id"]].drop_duplicates().merge(
        popularity,
        left_on="item_id",
        right_index=True,
        how="left",
    )

    bucket_df["interaction_count"] = bucket_df["interaction_count"].fillna(0)

    bucket_df = bucket_df.sort_values(
        ["interaction_count", "item_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    bucket_df["popularity_rank"] = bucket_df.index + 1

    if len(bucket_df) == 0:
        bucket_df["tail_bucket"] = []
        bucket_df["tail_item_score"] = []
        return bucket_df

    bucket_df["tail_bucket"] = (
        ((pd.Series(bucket_df.index, index=bucket_df.index) * num_buckets // len(bucket_df)) + 1)
        .clip(upper=num_buckets)
        .astype(int)
    )

    bucket_df["tail_item_score"] = 1 - (1 / bucket_df["tail_bucket"])

    return bucket_df[
        [
            "item_id",
            "interaction_count",
            "popularity_rank",
            "tail_bucket",
            "tail_item_score",
        ]
    ]


def tail_item_diversity_per_user_at_k(
    recommendations_k: pd.DataFrame,
    tail_scores: pd.DataFrame,
) -> pd.Series:
    recs = recommendations_k.merge(
        tail_scores[["item_id", "tail_item_score"]],
        on="item_id",
        how="left",
    )

    recs["tail_item_score"] = recs["tail_item_score"].fillna(0.0)

    return recs.groupby("user_id")["tail_item_score"].mean().rename(
        "tail_item_diversity"
    )


def head_tail_exposure_at_k(
    recommendations_k: pd.DataFrame,
    tail_scores: pd.DataFrame,
) -> pd.DataFrame:
    recs = recommendations_k.merge(
        tail_scores[["item_id", "tail_bucket"]],
        on="item_id",
        how="left",
    )

    recs["tail_bucket"] = recs["tail_bucket"].fillna(-1).astype(int)

    exposure = (
        recs.groupby("tail_bucket")
        .size()
        .rename("recommendation_count")
        .reset_index()
    )

    total = exposure["recommendation_count"].sum()
    exposure["recommendation_share"] = (
        exposure["recommendation_count"] / total if total else 0.0
    )

    return exposure


def welch_ttest_pvalue(a: pd.Series, b: pd.Series) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None

    try:
        from scipy.stats import ttest_ind
    except ImportError:
        return None

    return float(ttest_ind(a, b, equal_var=False).pvalue)


def build_analysis_tables(
    recommendations: pd.DataFrame,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    k: int,
    threshold: float,
    model_name: str,
    num_tail_buckets: int,
    threshold_sweep: list[float],
):
    recommendations_k = restrict_top_k(recommendations, k)

    diversity = brand_diversity_per_user_at_k(
        recommendations_k=recommendations_k,
        items=items,
    )

    groups = brand_loyal_groups(
        interactions=interactions,
        items=items,
        threshold=threshold,
    )

    tail_scores = item_tail_bucket_scores(
        interactions=interactions,
        items=items,
        num_buckets=num_tail_buckets,
    )

    tail_diversity = tail_item_diversity_per_user_at_k(
        recommendations_k=recommendations_k,
        tail_scores=tail_scores,
    )

    exposure = head_tail_exposure_at_k(
        recommendations_k=recommendations_k,
        tail_scores=tail_scores,
    )

    joined = pd.concat([diversity, tail_diversity, groups], axis=1).dropna()

    joined["brand_loyal"] = joined["brand_loyal"].astype(bool)
    loyal_scores = joined.loc[joined["brand_loyal"] == True, "brand_diversity"]
    non_loyal_scores = joined.loc[joined["brand_loyal"] == False, "brand_diversity"]

    loyal_mean = float(loyal_scores.mean()) if len(loyal_scores) else 0.0
    non_loyal_mean = float(non_loyal_scores.mean()) if len(non_loyal_scores) else 0.0

    metrics = {
        "model": model_name,
        f"catalog_coverage@{k}": catalog_coverage_at_k(recommendations_k, items),
        f"avg_brand_diversity@{k}": float(diversity.mean()) if len(diversity) else 0.0,
        f"avg_tail_item_diversity@{k}": float(tail_diversity.mean()) if len(tail_diversity) else 0.0,
        f"brand_loyal_avg_diversity@{k}": loyal_mean,
        f"non_brand_loyal_avg_diversity@{k}": non_loyal_mean,
        f"brand_diversity_gap_loyal_minus_non_loyal@{k}": loyal_mean - non_loyal_mean,
        "brand_loyal_users_n": int(len(loyal_scores)),
        "non_brand_loyal_users_n": int(len(non_loyal_scores)),
        "brand_diversity_welch_ttest_p_value": welch_ttest_pvalue(
            loyal_scores,
            non_loyal_scores,
        ),
    }

    per_user = joined.reset_index()
    per_user.insert(0, "model", model_name)
    per_user["user_group"] = per_user["brand_loyal"].map(
        {True: "brand-loyal", False: "non-brand-loyal"}
    )

    group_summary = (
        per_user.groupby("user_group")["brand_diversity"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    group_summary.insert(0, "model", model_name)
    group_summary["sem"] = group_summary["std"] / group_summary["count"].pow(0.5)

    threshold_rows = []

    for sweep_threshold in threshold_sweep:
        sweep_groups = brand_loyal_groups(
            interactions=interactions,
            items=items,
            threshold=sweep_threshold,
        )

        sweep_joined = pd.concat([diversity, sweep_groups], axis=1).dropna()

        sweep_joined["brand_loyal"] = sweep_joined["brand_loyal"].astype(bool)

        sweep_loyal = sweep_joined.loc[
            sweep_joined["brand_loyal"] == True,
            "brand_diversity",
        ]
        sweep_non_loyal = sweep_joined.loc[
            sweep_joined["brand_loyal"] == False,
            "brand_diversity",
        ]

        sweep_loyal_mean = float(sweep_loyal.mean()) if len(sweep_loyal) else 0.0
        sweep_non_loyal_mean = (
            float(sweep_non_loyal.mean()) if len(sweep_non_loyal) else 0.0
        )

        threshold_rows.append(
            {
                "model": model_name,
                "threshold": sweep_threshold,
                "brand_loyal_users_n": int(len(sweep_loyal)),
                "non_brand_loyal_users_n": int(len(sweep_non_loyal)),
                f"brand_loyal_avg_diversity@{k}": sweep_loyal_mean,
                f"non_brand_loyal_avg_diversity@{k}": sweep_non_loyal_mean,
                f"brand_diversity_gap_loyal_minus_non_loyal@{k}": sweep_loyal_mean
                - sweep_non_loyal_mean,
                "p_value": welch_ttest_pvalue(sweep_loyal, sweep_non_loyal),
            }
        )

    threshold_summary = pd.DataFrame(threshold_rows)

    return metrics, per_user, group_summary, threshold_summary, tail_scores, exposure


def write_plots(
    metrics: dict,
    per_user: pd.DataFrame,
    group_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    exposure: pd.DataFrame,
    output_dir: Path,
    k: int,
) -> None:
    plt = _safe_import_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(metrics.get("model", "model"))

    coverage_key = f"catalog_coverage@{k}"
    brand_key = f"avg_brand_diversity@{k}"
    tail_key = f"avg_tail_item_diversity@{k}"
    gap_key = f"brand_diversity_gap_loyal_minus_non_loyal@{k}"

    global_plot_data = pd.DataFrame(
        {
            "metric": [
                "Catalog Coverage",
                "Avg Brand Diversity",
                "Avg Tail Diversity",
            ],
            "value": [
                metrics[coverage_key],
                metrics[brand_key],
                metrics[tail_key],
            ],
        }
    )

    plt.figure(figsize=(8, 5))
    plt.bar(global_plot_data["metric"], global_plot_data["value"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{model_name}: diversity metrics@{k}")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"global_diversity_metrics_at_{k}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(
        group_summary["user_group"],
        group_summary["mean"],
        yerr=group_summary["sem"].fillna(0),
        capsize=4,
    )
    plt.ylim(0, 1)
    plt.ylabel(f"Brand Diversity@{k}")
    plt.title(f"{model_name}: brand diversity by user group@{k}")
    plt.tight_layout()
    plt.savefig(output_dir / f"brand_diversity_by_user_group_at_{k}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    box_groups = [
        per_user.loc[per_user["user_group"] == group, "brand_diversity"]
        for group in group_summary["user_group"]
    ]
    plt.boxplot(box_groups, labels=group_summary["user_group"], showmeans=True)
    plt.ylim(0, 1)
    plt.ylabel(f"Brand Diversity@{k}")
    plt.title(f"{model_name}: user-level brand diversity distribution@{k}")
    plt.tight_layout()
    plt.savefig(output_dir / f"brand_diversity_distribution_at_{k}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.axhline(0, color="black", linewidth=1)
    plt.bar(["Loyal - non-loyal"], [metrics[gap_key]])
    plt.ylabel(f"Brand Diversity Gap@{k}")
    plt.title(f"{model_name}: user-group diversity gap@{k}")
    plt.tight_layout()
    plt.savefig(output_dir / f"brand_diversity_group_gap_at_{k}.png", dpi=200)
    plt.close()

    threshold_gap_col = f"brand_diversity_gap_loyal_minus_non_loyal@{k}"

    plt.figure(figsize=(8, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.plot(
        threshold_summary["threshold"],
        threshold_summary[threshold_gap_col],
        marker="o",
    )
    plt.xlabel("Brand-loyal threshold")
    plt.ylabel(f"Loyal - non-loyal Brand Diversity@{k}")
    plt.title(f"{model_name}: brand-diversity gap across loyalty thresholds")
    plt.tight_layout()
    plt.savefig(output_dir / f"brand_loyal_threshold_sweep_gap_at_{k}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        exposure["tail_bucket"].astype(str),
        exposure["recommendation_share"],
    )
    plt.xlabel("Tail bucket")
    plt.ylabel("Recommendation share")
    plt.title(f"{model_name}: recommendation exposure by popularity bucket@{k}")
    plt.tight_layout()
    plt.savefig(output_dir / f"tail_bucket_exposure_at_{k}.png", dpi=200)
    plt.close()


def validate_required_columns(
    recommendations: pd.DataFrame,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
) -> None:
    required_recs = {"user_id", "item_id"}
    required_inter = {"user_id", "item_id"}
    required_items = {"item_id", "brand"}

    missing_recs = required_recs - set(recommendations.columns)
    missing_inter = required_inter - set(interactions.columns)
    missing_items = required_items - set(items.columns)

    if missing_recs:
        raise ValueError(f"Missing recommendation columns: {sorted(missing_recs)}")
    if missing_inter:
        raise ValueError(f"Missing interaction columns: {sorted(missing_inter)}")
    if missing_items:
        raise ValueError(f"Missing item columns: {sorted(missing_items)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendations", required=True, type=Path)
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--brand-loyal-threshold", type=float, default=0.5)
    parser.add_argument("--tail-buckets", type=int, default=10)
    parser.add_argument(
        "--brand-loyal-thresholds",
        default="0.2,0.3,0.4,0.5",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    recommendations = normalize_columns(read_table(args.recommendations, ","))
    interactions = normalize_columns(read_table(args.interactions, "\t"))
    items = normalize_columns(read_table(args.items, "\t"))

    validate_required_columns(recommendations, interactions, items)

    threshold_sweep = [
        float(value.strip())
        for value in args.brand_loyal_thresholds.split(",")
        if value.strip()
    ]

    (
        metrics,
        per_user,
        group_summary,
        threshold_summary,
        tail_scores,
        exposure,
    ) = build_analysis_tables(
        recommendations=recommendations,
        interactions=interactions,
        items=items,
        k=args.k,
        threshold=args.brand_loyal_threshold,
        model_name=args.model_name,
        num_tail_buckets=args.tail_buckets,
        threshold_sweep=threshold_sweep,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame([metrics])
    print(result.to_string(index=False))

    result.to_csv(args.output_dir / f"brand_metrics_summary_at_{args.k}.csv", index=False)
    group_summary.to_csv(
        args.output_dir / f"brand_diversity_group_summary_at_{args.k}.csv",
        index=False,
    )
    threshold_summary.to_csv(
        args.output_dir / f"brand_loyal_threshold_sweep_at_{args.k}.csv",
        index=False,
    )
    tail_scores.to_csv(
        args.output_dir / f"item_tail_buckets_at_{args.k}.csv",
        index=False,
    )
    exposure.to_csv(
        args.output_dir / f"tail_bucket_exposure_at_{args.k}.csv",
        index=False,
    )
    per_user.to_csv(
        args.output_dir / f"brand_diversity_per_user_at_{args.k}.csv",
        index=False,
    )

    notes = [
        f"Model: {args.model_name}",
        f"K: {args.k}",
        f"Brand-loyal threshold: {args.brand_loyal_threshold}",
        f"Tail popularity buckets: {args.tail_buckets}",
        "",
        "Catalog Coverage@K = unique recommended items / total catalog items.",
        "Brand Diversity@K = unique non-missing recommended brands / K recommended items.",
        "Missing brand metadata therefore reduces the diversity score and must be reported as a limitation.",
        "Brand-loyal users are users whose most frequent historical brand reaches the selected threshold.",
        "The brand-loyal split is exploratory because the Amazon Beauty metadata has many missing brands and the resulting groups are highly imbalanced.",
        "Tail Item Diversity@K is the average popularity-bucket score of recommended items.",
        "Tail bucket 1 is the most popular/head bucket; higher buckets are deeper-tail items.",
    ]

    (args.output_dir / f"brand_metrics_notes_at_{args.k}.txt").write_text(
        "\n".join(notes),
        encoding="utf-8",
    )

    if not args.no_plots:
        write_plots(
            metrics=metrics,
            per_user=per_user,
            group_summary=group_summary,
            threshold_summary=threshold_summary,
            exposure=exposure,
            output_dir=args.output_dir,
            k=args.k,
        )


if __name__ == "__main__":
    main()
