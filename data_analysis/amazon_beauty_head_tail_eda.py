from pathlib import Path
import argparse
import ast
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATASET_NAME = "amazon_beauty"


def unzip_if_needed(zip_path: Path, work_dir: Path) -> Path:
    dataset_dir = work_dir / zip_path.stem

    if dataset_dir.exists() and any(dataset_dir.rglob("*")):
        print(f"Using existing extracted folder: {dataset_dir}")
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {dataset_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    return dataset_dir


def find_file(dataset_dir: Path, suffix: str):
    candidates = sorted(dataset_dir.rglob(f"*{suffix}"))
    return candidates[0] if candidates else None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix in [".inter", ".item", ".user", ".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def normalize_recbole_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    RecBole columns often look like:
      user_id:token
      item_id:token
      rating:float
      timestamp:float

    This renames them to:
      user_id
      item_id
      rating
      timestamp
    """
    rename_map = {}
    for col in df.columns:
        clean = col.split(":")[0]
        rename_map[col] = clean
    return df.rename(columns=rename_map)


def load_interactions(dataset_dir: Path) -> pd.DataFrame:
    inter_file = find_file(dataset_dir, ".inter")

    if inter_file is None:
        raise FileNotFoundError(
            f"No .inter file found under {dataset_dir}. "
            "Please inspect the zip contents."
        )

    print(f"Loading interactions from: {inter_file}")
    df = read_table(inter_file)
    df = normalize_recbole_columns(df)

    required = {"user_id", "item_id"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Interaction file must contain user_id and item_id. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[["user_id", "item_id"]].dropna()
    return df


def load_item_metadata(dataset_dir: Path):
    item_file = find_file(dataset_dir, ".item")

    if item_file is None:
        print("No .item metadata file found. Skipping metadata-aware analysis.")
        return None

    print(f"Loading item metadata from: {item_file}")
    item_df = read_table(item_file)
    item_df = normalize_recbole_columns(item_df)

    if "item_id" not in item_df.columns:
        print(f"No item_id column found in item metadata. Columns: {item_df.columns.tolist()}")
        return None

    return item_df


def safe_parse_category(value):
    """
    Attempts to parse category-like values if stored as a list string.
    Otherwise returns the original value.
    """
    if pd.isna(value):
        return "UNKNOWN"

    if not isinstance(value, str):
        return str(value)

    stripped = value.strip()

    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, list):
            if len(parsed) == 0:
                return "UNKNOWN"
            return str(parsed[-1])
        return str(parsed)
    except Exception:
        return stripped if stripped else "UNKNOWN"


def maybe_numeric_price(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).replace("$", "").replace(",", "").strip()

    try:
        return float(value)
    except Exception:
        return np.nan


def make_basic_plots(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    item_counts = df["item_id"].value_counts()
    user_counts = df["user_id"].value_counts()

    num_interactions = len(df)
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()

    sorted_items = item_counts.sort_values(ascending=False)
    sorted_users = user_counts.sort_values(ascending=False)

    # Head = top 20% most popular items, tail = bottom 80%.
    n_head_top20 = max(1, int(np.ceil(0.20 * num_items)))
    head_top20_items = set(sorted_items.index[:n_head_top20])
    tail_bottom80_items = set(sorted_items.index[n_head_top20:])

    head_top20_interactions = df["item_id"].isin(head_top20_items).sum()
    tail_bottom80_interactions = df["item_id"].isin(tail_bottom80_items).sum()

    # Head = smallest set of popular items covering 80% of interactions.
    cumulative_interaction_share = sorted_items.cumsum() / sorted_items.sum()
    n_items_covering_80 = int((cumulative_interaction_share < 0.80).sum()) + 1

    head_cover80_items = set(sorted_items.index[:n_items_covering_80])
    tail_after80_items = set(sorted_items.index[n_items_covering_80:])

    head_cover80_interactions = df["item_id"].isin(head_cover80_items).sum()
    tail_after80_interactions = df["item_id"].isin(tail_after80_items).sum()

    summary = {
        "dataset": DATASET_NAME,
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,

        "item_popularity_min": int(item_counts.min()),
        "item_popularity_median": float(item_counts.median()),
        "item_popularity_mean": float(item_counts.mean()),
        "item_popularity_max": int(item_counts.max()),

        "user_activity_min": int(user_counts.min()),
        "user_activity_median": float(user_counts.median()),
        "user_activity_mean": float(user_counts.mean()),
        "user_activity_max": int(user_counts.max()),

        "top20_head_num_items": len(head_top20_items),
        "top20_tail_num_items": len(tail_bottom80_items),
        "top20_head_catalog_share": len(head_top20_items) / num_items,
        "top20_tail_catalog_share": len(tail_bottom80_items) / num_items,
        "top20_head_interaction_share": head_top20_interactions / num_interactions,
        "top20_tail_interaction_share": tail_bottom80_interactions / num_interactions,

        "cover80_head_num_items": len(head_cover80_items),
        "cover80_tail_num_items": len(tail_after80_items),
        "cover80_head_catalog_share": len(head_cover80_items) / num_items,
        "cover80_tail_catalog_share": len(tail_after80_items) / num_items,
        "cover80_head_interaction_share": head_cover80_interactions / num_interactions,
        "cover80_tail_interaction_share": tail_after80_interactions / num_interactions,
    }

    summary_df = pd.DataFrame([summary])

    
    item_popularity_df = pd.DataFrame({
        "item_id": sorted_items.index,
        "interaction_count": sorted_items.values,
        "popularity_rank": np.arange(1, num_items + 1),
        "is_head_top20": [item in head_top20_items for item in sorted_items.index],
        "is_tail_bottom80": [item in tail_bottom80_items for item in sorted_items.index],
        "is_head_cover80": [item in head_cover80_items for item in sorted_items.index],
        "is_tail_after80": [item in tail_after80_items for item in sorted_items.index],
    })

    item_popularity_df.to_csv(
        output_dir / "amazon_beauty_item_popularity_table.csv",
        index=False,
    )

    # Item popularity rank plot.
    plt.figure()
    plt.plot(np.arange(1, num_items + 1), sorted_items.values)
    plt.yscale("log")
    plt.xlabel("Item rank by popularity")
    plt.ylabel("Number of interactions (log scale)")
    plt.title("Amazon Beauty: Item popularity distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "amazon_beauty_item_popularity_log.png", dpi=200)
    plt.close()

    # Cumulative interaction share.
    plt.figure()
    plt.plot(
        np.arange(1, num_items + 1) / num_items,
        cumulative_interaction_share.values,
    )
    plt.axhline(0.80, linestyle="--", linewidth=1)
    plt.axvline(n_items_covering_80 / num_items, linestyle="--", linewidth=1)
    plt.xlabel("Fraction of items sorted by popularity")
    plt.ylabel("Cumulative fraction of interactions")
    plt.title("Amazon Beauty: Cumulative interaction share")
    plt.tight_layout()
    plt.savefig(output_dir / "amazon_beauty_cumulative_interactions.png", dpi=200)
    plt.close()

    # Item popularity histogram.
    plt.figure()
    plt.hist(item_counts.values, bins=50)
    plt.yscale("log")
    plt.xlabel("Interactions per item")
    plt.ylabel("Number of items (log scale)")
    plt.title("Amazon Beauty: Item popularity histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "amazon_beauty_item_popularity_hist.png", dpi=200)
    plt.close()

    # User activity rank plot.
    plt.figure()
    plt.plot(np.arange(1, num_users + 1), sorted_users.values)
    plt.yscale("log")
    plt.xlabel("User rank by activity")
    plt.ylabel("Number of interactions (log scale)")
    plt.title("Amazon Beauty: User activity distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "amazon_beauty_user_activity_log.png", dpi=200)
    plt.close()

    summary_df.to_csv(output_dir / "amazon_beauty_head_tail_summary.csv", index=False)

    print("\nBasic interaction summary:")
    print(summary_df.T)

    return item_popularity_df

def make_metadata_summaries(
    item_popularity_df: pd.DataFrame,
    item_metadata,
    output_dir: Path,
) -> None:
    if item_metadata is None:
        return

    merged = item_popularity_df.merge(item_metadata, on="item_id", how="left")
    merged.to_csv(output_dir / "amazon_beauty_item_popularity_with_metadata.csv", index=False)

    print("\nMetadata columns:")
    print(merged.columns.tolist())

    possible_brand_cols = [c for c in merged.columns if c.lower() in {"brand", "brand_name"}]
    possible_category_cols = [
        c for c in merged.columns
        if "categor" in c.lower() or c.lower() in {"class", "subcategory"}
    ]
    possible_price_cols = [c for c in merged.columns if "price" in c.lower()]

    # Brand analysis
    if possible_brand_cols:
        brand_col = possible_brand_cols[0]
        brand_summary = (
            merged.assign(brand=merged[brand_col].fillna("UNKNOWN"))
            .groupby("brand")
            .agg(
                num_items=("item_id", "count"),
                head_top20_items=("is_head_top20", "sum"),
                tail_bottom80_items=("is_tail_bottom80", "sum"),
                mean_interactions=("interaction_count", "mean"),
                total_interactions=("interaction_count", "sum"),
            )
            .sort_values("total_interactions", ascending=False)
            .reset_index()
        )
        brand_summary.to_csv(output_dir / "amazon_beauty_brand_head_tail_summary.csv", index=False)

    # Category analysis
    if possible_category_cols:
        category_col = possible_category_cols[0]
        category_summary = (
            merged.assign(category=merged[category_col].apply(safe_parse_category))
            .groupby("category")
            .agg(
                num_items=("item_id", "count"),
                head_top20_items=("is_head_top20", "sum"),
                tail_bottom80_items=("is_tail_bottom80", "sum"),
                mean_interactions=("interaction_count", "mean"),
                total_interactions=("interaction_count", "sum"),
            )
            .sort_values("total_interactions", ascending=False)
            .reset_index()
        )
        category_summary.to_csv(
            output_dir / "amazon_beauty_category_head_tail_summary.csv",
            index=False,
        )

        top_categories = category_summary.head(20)
        plt.figure()
        plt.barh(top_categories["category"].astype(str), top_categories["total_interactions"])
        plt.xlabel("Total interactions")
        plt.ylabel("Category")
        plt.title("Amazon Beauty: Top categories by interactions")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "amazon_beauty_top_categories.png", dpi=200)
        plt.close()

    # Price analysis
    if possible_price_cols:
        price_col = possible_price_cols[0]
        merged["price_numeric"] = merged[price_col].apply(maybe_numeric_price)

        price_summary = (
            merged.groupby("is_head_top20")
            .agg(
                num_items=("item_id", "count"),
                price_available=("price_numeric", lambda x: x.notna().sum()),
                price_mean=("price_numeric", "mean"),
                price_median=("price_numeric", "median"),
                price_min=("price_numeric", "min"),
                price_max=("price_numeric", "max"),
            )
            .reset_index()
        )
        price_summary.to_csv(output_dir / "amazon_beauty_price_head_tail_summary.csv", index=False)

        head_prices = merged.loc[merged["is_head_top20"], "price_numeric"].dropna()
        tail_prices = merged.loc[merged["is_tail_bottom80"], "price_numeric"].dropna()

        if len(head_prices) > 0 and len(tail_prices) > 0:
            plt.figure()
            plt.hist(head_prices, bins=50, alpha=0.6, label="Head top 20%")
            plt.hist(tail_prices, bins=50, alpha=0.6, label="Tail bottom 80%")
            plt.xlabel("Price")
            plt.ylabel("Number of items")
            plt.title("Amazon Beauty: Price distribution by head/tail")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "amazon_beauty_price_head_tail_hist.png", dpi=200)
            plt.close()

    print("\nMetadata-aware summaries saved if metadata columns were found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Amazon_Beauty.zip or extracted Amazon Beauty folder.",
    )
    parser.add_argument(
        "--work-dir",
        default="data_analysis/unzipped_data",
        help="Directory used for extracted data if input is a zip file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_analysis/plots",
        help="Directory where plots and summary CSV files will be saved.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.suffix == ".zip":
        dataset_dir = unzip_if_needed(input_path, work_dir)
    else:
        dataset_dir = input_path

    interactions = load_interactions(dataset_dir)
    item_metadata = load_item_metadata(dataset_dir)

    print("\nLoaded Amazon Beauty interactions:")
    print(f"Rows: {len(interactions):,}")
    print(f"Users: {interactions['user_id'].nunique():,}")
    print(f"Items: {interactions['item_id'].nunique():,}")
    print(interactions.head())

    item_popularity_df = make_basic_plots(interactions, output_dir)
    make_metadata_summaries(item_popularity_df, item_metadata, output_dir)

    print("\nDone.")
    print(f"Outputs are in: {output_dir}")


if __name__ == "__main__":
    main()