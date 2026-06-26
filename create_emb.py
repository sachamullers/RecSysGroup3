from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

base_dir = Path("/home/scur1242/RecSysGroup3/dataset") #Change to own directory

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

#go over all .item files in the datasets directory
for item_path in base_dir.glob("*/*-without-emb.item"):
    output_name = item_path.name.replace("-without-emb.item", ".item")
    output_path = item_path.with_name(output_name)

    # Skip if embedded file already exists
    if output_path.exists():
        print(f"Skipping, already exists: {output_path}")
        continue

    print(f"Processing: {item_path}")

    df = pd.read_csv(
        item_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    # Take all columns except item_id to use for embeddings
    cols_to_embed = [
        col for col in df.columns
        if col not in ["item_id:token"]
    ]

    # Concatenate them into one text string per item
    texts = (
        df[cols_to_embed]
        .astype(str)
        .apply(lambda row: " ".join(x.strip() for x in row if x.strip() != ""), axis=1)
        .tolist()
    )

    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Save embeddings as space-separated float sequences
    df["llm_embedding:float_seq"] = [
        " ".join(map(str, emb)) for emb in embeddings
    ]

    # Keep only item, title and llm_embedding
    df_out = df[
        [
            "item_id:token",
            "title:token",
            "llm_embedding:float_seq"
        ]
    ]

    df_out.to_csv(output_path, sep="\t", index=False)