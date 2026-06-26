import argparse
import os
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------

DEFAULT_EXCEL_DIR = Path("/projects/prjs2120/groups/group_03/results/multiple_seeds")

BACKBONES = {"lightgcn", "sgl", "sgcl"}
VARIANTS = {"rand", "uni", "var"}


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def paired_one_tailed_ttest(a, b, alternative="greater"):
    """
    Tests whether mean(a) > mean(b), paired over seeds.
    """
    result = ttest_rel(a, b, alternative=alternative)

    if pd.isna(result.statistic) or pd.isna(result.pvalue):
        return 0.0, 1.0

    return result.statistic, result.pvalue


# ---------------------------------------------------------------------
# Excel reading
# ---------------------------------------------------------------------

def read_excel_results(excel_path, dataset):
    """
    Reads one Excel file for one dataset.

    Expected Excel format:

        A1: dataset name
        Row 2: metric names, e.g. recall@10 ... ndcg@10 ...
        Row 3: seeds, e.g. 2020 2021 2022 2023 2024
        Rows below:
            lightgcn
            rand
            uni
            var
            sgl
            rand
            uni
            var
            sgcl
            rand
            uni
            var

    Returns:
        data[model][variant][metric] = list of seed scores
    """

    df = pd.read_excel(excel_path, header=None)

    dataset_cell = str(df.iloc[0, 0]).strip().lower()
    if dataset.lower() not in dataset_cell:
        print(f"Warning: A1 says '{df.iloc[0, 0]}', but you requested '{dataset}'.")

    metric_row = df.iloc[1].ffill()
    seed_row = df.iloc[2]

    value_cols = list(range(1, df.shape[1]))

    metrics_by_col = {
        col: str(metric_row.iloc[col]).strip().lower()
        for col in value_cols
        if not pd.isna(metric_row.iloc[col])
    }

    seeds_by_col = {
        col: int(seed_row.iloc[col])
        for col in value_cols
        if not pd.isna(seed_row.iloc[col])
    }

    metrics = []
    for col in value_cols:
        metric = metrics_by_col.get(col)
        if metric and metric not in metrics:
            metrics.append(metric)

    data = {}
    current_model = None

    for row_idx in range(3, len(df)):
        label_raw = df.iloc[row_idx, 0]

        if pd.isna(label_raw):
            continue

        label = str(label_raw).strip().lower()

        if label in BACKBONES:
            current_model = label
            data[current_model] = {}
            variant_name = "baseline"

        elif label in VARIANTS:
            if current_model is None:
                raise ValueError(f"Variant '{label}' appears before any model row.")
            variant_name = label

        else:
            print(f"Skipping unknown row label: {label}")
            continue

        data[current_model][variant_name] = {}

        for metric in metrics:
            cols_for_metric = [
                col for col in value_cols
                if metrics_by_col.get(col) == metric
            ]

            scores = []

            for col in cols_for_metric:
                value = df.iloc[row_idx, col]

                if pd.isna(value):
                    raise ValueError(
                        f"Missing value at row {row_idx + 1}, column {col + 1}, "
                        f"model={current_model}, variant={variant_name}, metric={metric}"
                    )

                scores.append(float(value))

            data[current_model][variant_name][metric] = scores

    return data, metrics, list(seeds_by_col.values())


# ---------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------

def run_significance_tests(data, metrics, dataset):
    rows = []

    for model, model_data in data.items():
        if "baseline" not in model_data:
            raise ValueError(f"Missing baseline row for model '{model}'.")

        for metric in metrics:
            baseline = model_data["baseline"][metric]

            # ---------------------------------------------------------
            # 1. Rand / Uni / Var vs baseline
            # ---------------------------------------------------------

            pvals_baseline = []
            baseline_rows = []

            for variant in ["rand", "uni", "var"]:
                if variant not in model_data:
                    raise ValueError(f"Missing variant '{variant}' for model '{model}'.")

                scores = model_data[variant][metric]

                t_stat, p_value = paired_one_tailed_ttest(
                    scores,
                    baseline,
                    alternative="greater",
                )

                baseline_mean = pd.Series(baseline).mean()
                method_mean = pd.Series(scores).mean()

                improvement = (
                    (method_mean - baseline_mean)
                    / baseline_mean
                    * 100
                )

                row = {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "comparison_type": "variant_vs_baseline",
                    "comparison": f"{variant} > baseline",
                    "baseline_mean": baseline_mean,
                    "method_mean": method_mean,
                    "improvement_%": improvement,
                    "t_stat": t_stat,
                    "raw_p": p_value,
                }

                baseline_rows.append(row)
                pvals_baseline.append(p_value)

            reject, p_holm, _, _ = multipletests(
                pvals_baseline,
                alpha=0.05,
                method="holm",
            )

            for row, corrected_p, significant in zip(baseline_rows, p_holm, reject):
                row["holm_p"] = corrected_p
                row["significant_0.05"] = significant
                rows.append(row)

            # ---------------------------------------------------------
            # 2. Var vs Rand / Uni
            # ---------------------------------------------------------

            pvals_var = []
            var_rows = []

            for other in ["rand", "uni"]:
                var_scores = model_data["var"][metric]
                other_scores = model_data[other][metric]

                t_stat, p_value = paired_one_tailed_ttest(
                    var_scores,
                    other_scores,
                    alternative="greater",
                )

                other_mean = pd.Series(other_scores).mean()
                var_mean = pd.Series(var_scores).mean()

                improvement = (
                    (var_mean - other_mean)
                    / other_mean
                    * 100
                )

                row = {
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "comparison_type": "var_vs_other",
                    "comparison": f"var > {other}",
                    "baseline_mean": other_mean,
                    "method_mean": var_mean,
                    "improvement_%": improvement,
                    "t_stat": t_stat,
                    "raw_p": p_value,
                }

                var_rows.append(row)
                pvals_var.append(p_value)

            reject, p_holm, _, _ = multipletests(
                pvals_var,
                alpha=0.05,
                method="holm",
            )

            for row, corrected_p, significant in zip(var_rows, p_holm, reject):
                row["holm_p"] = corrected_p
                row["significant_0.05"] = significant
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run paired one-tailed t-tests with Holm correction from Excel seed results."
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Dataset name, e.g. office-products, amazon-beauty, toys-games, tools-home.",
    )

    parser.add_argument(
        "-x",
        "--excel",
        default=None,
        help=(
            "Path to Excel file. If omitted, the script uses "
            f"{DEFAULT_EXCEL_DIR}/<dataset>.xlsx."
        ),
    )

    args = parser.parse_args()

    # Use explicit Excel path if given, otherwise use default folder + dataset name
    if args.excel is None:
        excel_path = DEFAULT_EXCEL_DIR / f"{args.dataset}.xlsx"
    else:
        excel_path = Path(args.excel)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    data, metrics, seeds = read_excel_results(excel_path, args.dataset)
    results = run_significance_tests(data, metrics, args.dataset)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("Current working directory:", os.getcwd())
    print("Excel file:", excel_path)
    print("Detected metrics:", metrics)
    print("Detected seeds:", seeds)
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()