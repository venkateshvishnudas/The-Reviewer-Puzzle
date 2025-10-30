"""
build_embeddings.py
===================

Create and cache sentence embeddings for all parsed papers
using a pretrained embedding model (default: E5).

Run:
  python src/build_embeddings.py --model intfloat/e5-base-v2 --batch 32
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ---------------------------------------------------------------------
# Global Configs
# ---------------------------------------------------------------------
DATA_DIR = Path("data/cache")
CORPUS_PKL = DATA_DIR / "parsed_corpus.pkl"
EMB_ROOT = DATA_DIR / "embeddings"

# Force CPU to avoid DLL/driver issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def clean_text(s: str) -> str:
    """Clean up unwanted whitespace and special characters."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ")  # replace non-breaking space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine title, abstract, and body text for embedding."""
    joined = (
        df["title"].fillna("") + "\n\n" +
        df["abstract"].fillna("") + "\n\n" +
        df["body"].fillna("").where(df["body"].str.len().fillna(0) > 100, "") + "\n\n" +
        df["text"].fillna("")
    )
    return joined.apply(clean_text)


def encode_in_batches(model: SentenceTransformer, texts, batch_size=32, normalize=True):
    """Encode texts in manageable batches using SentenceTransformer."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔢 Embedding"):
        batch = texts[i:i + batch_size]
        batch = [f"passage: {t}" for t in batch]  # E5 model prompt format
        e = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        embs.append(e)
    return np.vstack(embs)


def check_numpy_version():
    """Warn user if NumPy 2.x is installed (may cause Torch errors)."""
    import numpy
    ver = tuple(map(int, numpy.__version__.split(".")[:2]))
    if ver[0] >= 2:
        print(f"⚠️  Warning: NumPy {numpy.__version__} detected. "
              f"Consider `pip install 'numpy<2'` for PyTorch 2.2 compatibility.")


# ---------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build embeddings for parsed research papers.")
    ap.add_argument("--model", default="intfloat/e5-base-v2",
                    help="HuggingFace model ID (e.g., intfloat/e5-base-v2, intfloat/e5-large-v2)")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for encoding")
    ap.add_argument("--outfile", default=None, help="Optional path override for output .npy file")
    args = ap.parse_args()

    # Check dataset
    assert CORPUS_PKL.exists(), f"❌ Missing corpus file: {CORPUS_PKL}. Run parsing step first."

    print("📘 Loading corpus...")
    df = pd.read_pickle(CORPUS_PKL)
    df["combined_text"] = build_text_field(df)
    df = df[df["combined_text"].str.len() > 50].reset_index(drop=True)
    print(f"✅ Loaded {len(df)} valid documents for embedding.\n")

    # Check NumPy version compatibility
    check_numpy_version()

    # Detect available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Using device: {device.upper()}")

    # Load model
    print(f"📦 Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    # Encode all documents
    print("🚀 Generating embeddings...")
    embs = encode_in_batches(model, df["combined_text"].tolist(), batch_size=args.batch, normalize=True)

    # Save artifacts
    model_slug = args.model.replace("/", "__")
    out_dir = EMB_ROOT / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = Path(args.outfile) if args.outfile else (out_dir / "doc_embeddings.npy")
    np.save(emb_path, embs)

    df_out = df[["author_id", "paper_id", "title", "abstract", "path"]].copy()
    df_out.to_pickle(out_dir / "embeddings_df.pkl")

    meta = {
        "model": args.model,
        "count": len(df_out),
        "dim": int(embs.shape[1]),
        "file": str(emb_path),
    }
    pd.Series(meta).to_json(out_dir / "meta.json", indent=2)

    print(f"\n✅ Saved {len(df_out)} embeddings @ {emb_path}")
    print(f"   📏 Dim: {embs.shape[1]} | 🧠 Model: {args.model}")
    print(f"   📂 Index DF: {out_dir / 'embeddings_df.pkl'}")
    print(f"   🗂️  Meta:     {out_dir / 'meta.json'}\n")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
