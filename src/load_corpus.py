from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import csv
import os
from functools import lru_cache

# -----------------------------------
# CONFIG
# -----------------------------------
META_FILE = Path("data/cache/parsed/meta_log.csv")
PARSED_DIR = Path("data/cache/parsed")
EXPECTED_COLS = ["author_id", "paper_id", "pdf_path", "json_path", "status"]

# -----------------------------------
# SAFE CSV LOADER
# -----------------------------------
def load_meta(meta_path: Path) -> pd.DataFrame:
    """Load meta_log.csv safely even if it's malformed."""
    try:
        df = pd.read_csv(meta_path, on_bad_lines="skip", engine="python")
        print(f"✅ Loaded meta_log.csv with {len(df)} entries")
    except pd.errors.ParserError:
        print("⚠️ ParserError: trying relaxed CSV mode...")
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample)
            df = pd.read_csv(meta_path, delimiter=dialect.delimiter, on_bad_lines="skip", engine="python")
        except Exception:
            df = pd.read_csv(meta_path, delimiter=",", on_bad_lines="skip", engine="python")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # PERF: Combined column filtering and NaN dropping
    df = df[[c for c in df.columns if c in EXPECTED_COLS]].dropna(subset=["json_path"])
    return df

# -----------------------------------
# MAIN LOADER
# -----------------------------------
def load_parsed_corpus(meta_path=META_FILE):
    """Load successfully parsed papers into a structured DataFrame."""
    df_meta = load_meta(meta_path)

    # PERF: Preprocess 'status' column to lowercase once
    df_meta["status"] = df_meta["status"].str.lower()
    df_meta = df_meta[df_meta["status"].str.contains("success", na=False)]

    # Normalize Windows paths to POSIX (forward slashes)
    df_meta["json_path"] = df_meta["json_path"].str.replace("\\", "/", regex=False)

    base = PARSED_DIR
    records = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Loading parsed JSONs"):
        json_path = Path(str(row["json_path"]).strip())

        # PERF: Store the result of the first existence check
        json_exists = json_path.exists()
        if not json_exists:
            alt_path = base / json_path.name
            if alt_path.exists():
                json_path = alt_path
            else:
                print(f"⚠️ Missing JSON file for {row.get('paper_id', '?')}: {json_path}")
                continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # PERF: Use str.join() for concatenation
            text = " ".join(filter(None, [data.get("title", ""), data.get("abstract", ""), data.get("body", "")])).strip()

            if len(text) < 100:
                continue  # skip tiny documents

            records.append({
                "author_id": row.get("author_id", ""),
                "paper_id": row.get("paper_id", json_path.stem),
                "title": data.get("title", ""),
                "abstract": data.get("abstract", ""),
                "body": data.get("body", ""),
                "text": text,
                "path": str(json_path)
            })

        except Exception as e:
            print(f"❌ Error reading {json_path.name}: {e}")

    # PERF: Use pd.DataFrame.from_records() for better performance
    df = pd.DataFrame.from_records(records)
    print(f"\n✅ Loaded {len(df)} valid parsed papers out of {len(df_meta)}")
    return df

# -----------------------------------
# RUN MODULE
# -----------------------------------
if __name__ == "__main__":
    corpus_df = load_parsed_corpus()

    if not corpus_df.empty:
        print("\n📘 Corpus Preview:")
        print(corpus_df.sample(min(5, len(corpus_df)))[["author_id", "paper_id", "title"]])
    else:
        print("\n⚠️ No valid parsed papers found! Check your meta_log.csv paths or parsing status.")