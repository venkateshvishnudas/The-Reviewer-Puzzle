"""
find_reviewers_semantic.py
==========================

Semantic Reviewer Recommendation using SentenceTransformer (E5).

Given a research paper (PDF or raw text), this module identifies
the top-k authors whose prior publications are semantically closest
to the input paper using cosine similarity on sentence embeddings.

Usage:
------
python src/find_reviewers_semantic.py --file path/to/paper.pdf
or
python src/find_reviewers_semantic.py --text "Transformer-based text classification"
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import argparse
import re
import torch
import os

# ---------------------------------------------------------------------
# 📁 Configuration
# ---------------------------------------------------------------------
EMB_ROOT = Path("data/cache/embeddings/intfloat__e5-base-v2")
EMB_PATH = EMB_ROOT / "doc_embeddings.npy"
META_PATH = EMB_ROOT / "embeddings_df.pkl"

# Force CPU mode unless GPU available
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------------
# 🧹 Utilities
# ---------------------------------------------------------------------
def clean_text(s: str) -> str:
    """Normalize whitespace and remove invisible characters."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_text_from_pdf(pdf_path: str | Path, max_chars: int = 4000) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return clean_text(text[:max_chars])
    except Exception as e:
        print(f"⚠️ Could not extract text from PDF: {e}")
        return ""


# ---------------------------------------------------------------------
# 🎯 Core Semantic Reviewer Finder
# ---------------------------------------------------------------------
def find_top_reviewers(query_input, top_k: int = 5) -> pd.DataFrame:
    """
    Finds top-k reviewer authors using semantic similarity (E5 embeddings).

    Args:
        query_input (str or Path): PDF file path or text of the research paper.
        top_k (int): Number of top reviewers to return.

    Returns:
        pd.DataFrame: DataFrame with 'author_id' and 'similarity' columns.
    """
    # -------------------------------------------------------------
    # 1️⃣ Input Handling
    # -------------------------------------------------------------
    if isinstance(query_input, (str, Path)) and str(query_input).lower().endswith(".pdf"):
        print(f"📄 Extracting text from PDF: {query_input}")
        query_text = extract_text_from_pdf(query_input)
    else:
        query_text = str(query_input)

    if len(query_text.strip()) < 50:
        raise ValueError("❌ Query too short. Please provide a full abstract or paper content.")

    # -------------------------------------------------------------
    # 2️⃣ Load Prebuilt Corpus Embeddings
    # -------------------------------------------------------------
    if not EMB_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("❌ Missing embeddings. Run build_embeddings.py first.")

    print("📦 Loading corpus embeddings...")
    corpus_embs = np.load(EMB_PATH)
    meta_df = pd.read_pickle(META_PATH)
    print(f"✅ Loaded {len(meta_df)} papers from {meta_df['author_id'].nunique()} authors.\n")

    # -------------------------------------------------------------
    # 3️⃣ Load SentenceTransformer (E5-base)
    # -------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ Using device: {device.upper()}")
    model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    # -------------------------------------------------------------
    # 4️⃣ Encode Query & Compute Similarities
    # -------------------------------------------------------------
    print("🚀 Encoding query...")
    query_emb = model.encode([f"query: {query_text}"], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, corpus_embs).flatten()

    # -------------------------------------------------------------
    # 5️⃣ Aggregate Scores per Author
    # -------------------------------------------------------------
    meta_df["similarity"] = sims
    ranked = (
        meta_df.groupby("author_id")["similarity"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
    )

    results = pd.DataFrame({
        "author_id": ranked.index,
        "similarity": ranked.values
    }).reset_index(drop=True)

    return results


# ---------------------------------------------------------------------
# 🧪 CLI Entry (for quick testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic reviewer recommendation using E5 embeddings.")
    parser.add_argument("--file", type=str, help="Path to research paper (PDF).")
    parser.add_argument("--text", type=str, help="Raw text or abstract of the paper.")
    parser.add_argument("--topk", type=int, default=5, help="Number of reviewers to return.")
    args = parser.parse_args()

    query_input = args.file if args.file else args.text
    if not query_input:
        print("⚠️ Please provide either --file or --text argument.")
        exit(1)

    results = find_top_reviewers(query_input, top_k=args.topk)

    print("\n🎯 Top Recommended Reviewers:")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['author_id']:<25} | Similarity: {row['similarity']:.4f}")

    print("\n✅ Semantic reviewer matching complete.")
