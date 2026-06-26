import pandas as pd

plain_path = "logs_connector/phase2/beauty_low5_contgcn_var_noresid_run2_summary.csv"
resid_path = "logs_connector/phase2/beauty_low5_contgcn_var_resid_run2_summary.csv"

plain = pd.read_csv(plain_path)
resid = pd.read_csv(resid_path)

def get_row(df, mode):
    row = df[df["mode"] == mode]
    if row.empty:
        raise ValueError(f"Mode {mode} not found")
    return row.iloc[0]

random_row = get_row(plain, "random")
direct_row = get_row(plain, "direct")
plain_conn_row = get_row(plain, "connector")
resid_conn_row = get_row(resid, "connector")

rows = [
    ("LightGCN", random_row),
    ("LLMInit-Var", direct_row),
    ("Plain connector", plain_conn_row),
    (r"Residual connector, $\alpha = 0.1$", resid_conn_row),
]

print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Cold-start performance of connector variants on the Beauty Low-5 split. We report performance on cold items only and when cold items are ranked together with seen items.}")
print(r"\label{tab:connector_low5}")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"& \multicolumn{2}{c}{Cold only} & \multicolumn{2}{c}{Seen + cold} \\")
print(r"Method & R@10 & N@10 & R@10 & N@10 \\")
print(r"\midrule")

for name, row in rows:
    print(
        f"{name} & "
        f"{row['removed_only_recall@10']:.4f} & "
        f"{row['removed_only_ndcg@10']:.4f} & "
        f"{row['seen_plus_removed_recall@10']:.4f} & "
        f"{row['seen_plus_removed_ndcg@10']:.4f} \\\\"
    )

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
