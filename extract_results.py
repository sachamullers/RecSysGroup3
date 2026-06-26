import ast
import glob
import os
import re
import pandas as pd

LOG_DIR = "/home/scur1242/RecSysGroup3/logsfinal"

# Map array task IDs to dataset/method labels for each block
TASK_MAP = {
    "block_lightgcn": {
        0: ("Tools--Home", "LightGCN", "baseline"),
        1: ("Tools--Home", "+LLMInit--Rand", "rand"),
        2: ("Tools--Home", "+LLMInit--Uni", "uni"),
        3: ("Tools--Home", "+LLMInit--Var", "var"),
        4: ("Office--Products", "LightGCN", "baseline"),
        5: ("Office--Products", "+LLMInit--Rand", "rand"),
        6: ("Office--Products", "+LLMInit--Uni", "uni"),
        7: ("Office--Products", "+LLMInit--Var", "var"),
    },
    "block_sgl": {
        0: ("Tools--Home", "SGL", "baseline"),
        1: ("Tools--Home", "+LLMInit--Rand", "rand"),
        2: ("Tools--Home", "+LLMInit--Uni", "uni"),
        3: ("Tools--Home", "+LLMInit--Var", "var"),
        4: ("Office--Products", "SGL", "baseline"),
        5: ("Office--Products", "+LLMInit--Rand", "rand"),
        6: ("Office--Products", "+LLMInit--Uni", "uni"),
        7: ("Office--Products", "+LLMInit--Var", "var"),
    },
    "block_sgcl": {
        0: ("Tools--Home", "SGCL", "baseline"),
        1: ("Tools--Home", "+LLMInit--Rand", "rand"),
        2: ("Tools--Home", "+LLMInit--Uni", "uni"),
        3: ("Tools--Home", "+LLMInit--Var", "var"),
        4: ("Office--Products", "SGCL", "baseline"),
        5: ("Office--Products", "+LLMInit--Rand", "rand"),
        6: ("Office--Products", "+LLMInit--Uni", "uni"),
        7: ("Office--Products", "+LLMInit--Var", "var"),
    },
}


def parse_metrics(text):
    matches = re.findall(r"test result: OrderedDict\((\[.*?\])\)", text)
    if not matches:
        return None

    pairs = ast.literal_eval(matches[-1])
    metrics = dict(pairs)

    return {
        "recall@10": metrics.get("recall@10"),
        "ndcg@10": metrics.get("ndcg@10"),
        "recall@20": metrics.get("recall@20"),
        "ndcg@20": metrics.get("ndcg@20"),
        "recall@50": metrics.get("recall@50"),
        "ndcg@50": metrics.get("ndcg@50"),
    }


def detect_block_and_task(path):
    name = os.path.basename(path)

    m = re.match(r"(block_[a-zA-Z0-9]+)-(\d+)_(\d+)\.err", name)
    if not m:
        return None, None, None

    block = m.group(1)
    jobid = m.group(2)
    task = int(m.group(3))

    return block, jobid, task


rows = []

for path in glob.glob(os.path.join(LOG_DIR, "block_*.err")):
    block, jobid, task = detect_block_and_task(path)

    if block not in TASK_MAP:
        continue

    if task not in TASK_MAP[block]:
        continue

    with open(path, "r", errors="ignore") as f:
        text = f.read()

    metrics = parse_metrics(text)
    if metrics is None:
        continue

    dataset, method, variant = TASK_MAP[block][task]

    # Group tells us which section of the paper table this row belongs to
    if block == "block_lightgcn":
        group = "LightGCN"
    elif block == "block_sgl":
        group = "SGL"
    elif block == "block_sgcl":
        group = "SGCL"
    else:
        group = block

    command_match = re.search(r"Command:\s*(.*)", text)
    command = command_match.group(1).strip() if command_match else ""

    rows.append({
        "block": block,
        "jobid": jobid,
        "task": task,
        "group": group,
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "recall@10": metrics["recall@10"],
        "ndcg@10": metrics["ndcg@10"],
        "recall@20": metrics["recall@20"],
        "ndcg@20": metrics["ndcg@20"],
        "recall@50": metrics["recall@50"],
        "ndcg@50": metrics["ndcg@50"],
        "file": path,
        "command": command,
    })


df = pd.DataFrame(rows)

if df.empty:
    print("No completed test results found.")
    raise SystemExit(1)

# If multiple runs exist for same group/dataset/method, keep latest by jobid/task file order
df = df.sort_values(["group", "dataset", "method", "jobid", "task"])
df = df.drop_duplicates(subset=["group", "dataset", "method"], keep="last")

df = df.sort_values(["group", "dataset", "task"])
df.to_csv("extracted_reproduction_results.csv", index=False)

print("\nSaved CSV:")
print("extracted_reproduction_results.csv")

print("\nCompact results:")
print(df[["group", "dataset", "method", "recall@10", "ndcg@10", "file"]].to_string(index=False))


def fmt(x):
    if pd.isna(x):
        return "--"
    return f"{x:.4f}"


def imp(value, base):
    if pd.isna(value) or pd.isna(base) or base == 0:
        return "--"
    pct = ((value - base) / base) * 100
    color = "impgreen" if pct >= 0 else "red"
    sign = "+" if pct >= 0 else ""
    return rf"\textcolor{{{color}}}{{{sign}{pct:.1f}\%}}"


print("\n\nLaTeX cells for Tools--Home and Office--Products")
print("Use these to replace the -- & -- & -- & -- parts in your table.\n")

for group in ["LightGCN", "SGL", "SGCL"]:
    print(f"% ================= {group} block =================")

    gdf = df[df["group"] == group]

    base_tools = gdf[(gdf["dataset"] == "Tools--Home") & (gdf["method"] == group)]
    base_office = gdf[(gdf["dataset"] == "Office--Products") & (gdf["method"] == group)]

    bt_r = base_tools["recall@10"].iloc[0] if len(base_tools) else float("nan")
    bt_n = base_tools["ndcg@10"].iloc[0] if len(base_tools) else float("nan")
    bo_r = base_office["recall@10"].iloc[0] if len(base_office) else float("nan")
    bo_n = base_office["ndcg@10"].iloc[0] if len(base_office) else float("nan")

    print(f"% {group}")
    print(f"& {fmt(bt_r)} & {fmt(bt_n)} & {fmt(bo_r)} & {fmt(bo_n)} \\\\")

    for method in ["+LLMInit--Rand", "+LLMInit--Uni", "+LLMInit--Var"]:
        tools = gdf[(gdf["dataset"] == "Tools--Home") & (gdf["method"] == method)]
        office = gdf[(gdf["dataset"] == "Office--Products") & (gdf["method"] == method)]

        tr = tools["recall@10"].iloc[0] if len(tools) else float("nan")
        tn = tools["ndcg@10"].iloc[0] if len(tools) else float("nan")
        orr = office["recall@10"].iloc[0] if len(office) else float("nan")
        on = office["ndcg@10"].iloc[0] if len(office) else float("nan")

        print(f"% {method}")
        print(f"& {fmt(tr)} & {fmt(tn)} & {fmt(orr)} & {fmt(on)} \\\\")
        print(f"& {imp(tr, bt_r)} & {imp(tn, bt_n)} & {imp(orr, bo_r)} & {imp(on, bo_n)} \\\\")

    print()
