from pathlib import Path
import torch
import pandas as pd


SAVED_DIR = Path("saved")
OUT_PATH = Path("logs_connector/checkpoint_inventory.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_config_value(config, key, default=""):
    try:
        if key in config:
            return config[key]
    except Exception:
        pass
    return default


def inspect_checkpoint(path):
    row = {
        "checkpoint_path": str(path),
        "filename": path.name,
        "model": "",
        "dataset": "",
        "emb_selection": "",
        "use_connector": "",
        "use_residual_connector": "",
        "residual_connector_alpha": "",
        "connector_align_weight": "",
        "embedding_size": "",
        "llm_embedding_size": "",
        "has_connector_weights": False,
        "connector_weight_keys": "",
        "notes": "",
    }

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        row["notes"] = f"FAILED_TO_LOAD: {e}"
        return row

    config = ckpt.get("config", None)

    if config is not None:
        for key in [
            "model",
            "dataset",
            "emb_selection",
            "use_connector",
            "use_residual_connector",
            "residual_connector_alpha",
            "connector_align_weight",
            "embedding_size",
            "llm_embedding_size",
        ]:
            row[key] = get_config_value(config, key, "")

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))

    if state is not None:
        connector_keys = [k for k in state.keys() if "connector" in k.lower()]
        row["has_connector_weights"] = len(connector_keys) > 0
        row["connector_weight_keys"] = "; ".join(connector_keys)

    if row["use_connector"] == "":
        row["notes"] = "use_connector not found in config"

    return row


def main():
    checkpoints = sorted(SAVED_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime)

    rows = [inspect_checkpoint(p) for p in checkpoints]
    df = pd.DataFrame(rows)

    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote: {OUT_PATH}")
    print()
    print("Recent checkpoints:")
    cols = [
        "checkpoint_path",
        "model",
        "dataset",
        "emb_selection",
        "use_connector",
        "use_residual_connector",
        "residual_connector_alpha",
        "has_connector_weights",
    ]
    print(df[cols].tail(30).to_string(index=False))


if __name__ == "__main__":
    main()
