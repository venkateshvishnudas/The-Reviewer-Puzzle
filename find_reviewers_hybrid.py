"""
find_reviewers_hybrid.py (v3.0)
===============================

Hybrid Reviewer Recommendation System
-------------------------------------
Combines lexical (TF-IDF) and semantic (E5-base) similarities
for balanced reviewer matching.

Formula:
    final_score = α * TFIDF_similarity + (1 - α) * Semantic_similarity

Features:
- Strict data alignment via paper_id
- Safe normalization with epsilon
- Max aggregation per author (strongest topical match)
- Hyphen/number-preserving cleaner
- CPU-safe, Streamlit-ready, deployment-stable
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import argparse
import re
import os
import torch

# -------------------------------------------------------------------
# 📁 Configuration
# -------------------------------------------------------------------
TFIDF_DIR = Path("data/cache/tfidf_index")
EMB_DIR = Path("data/cache/embeddings/intfloat__e5-base-v2")

# CPU fallback for deployment safety
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -------------------------------------------------------------------
# 🧹 Utility Functions
# -------------------------------------------------------------------
def clean_text(s: str) -> str:
    """Clean text but keep scientific tokens (hyphens, numbers)."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[^a-zA-Z0-9\s\-]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_text_from_pdf(pdf_path: str | Path, max_chars: int = 4000) -> str:
    """Extract readable text from a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return clean_text(text[:max_chars])
    except Exception as e:
        print(f"⚠️ Could not extract text from PDF: {e}")
        return ""


# -------------------------------------------------------------------
# ⚙️ Load Models and Data
# -------------------------------------------------------------------
print("📦 Loading TF-IDF and Semantic Embedding models...")

vectorizer = joblib.load(TFIDF_DIR / "vectorizer.pkl")
tfidf_matrix = joblib.load(TFIDF_DIR / "tfidf_matrix.pkl")
tfidf_df = pd.read_pickle(TFIDF_DIR / "tfidf_df.pkl")

corpus_embs = np.load(EMB_DIR / "doc_embeddings.npy")
emb_df = pd.read_pickle(EMB_DIR / "embeddings_df.pkl")

# 🧩 Align TF-IDF and Embedding Data
if "paper_id" in tfidf_df.columns and "paper_id" in emb_df.columns:
    common_ids = sorted(set(tfidf_df["paper_id"]).intersection(set(emb_df["paper_id"])))
    if not common_ids:
        raise ValueError("❌ No common paper_ids found between TF-IDF and Embeddings datasets.")
    tfidf_df = tfidf_df[tfidf_df["paper_id"].isin(common_ids)].reset_index(drop=True)
    emb_df = emb_df[emb_df["paper_id"].isin(common_ids)].reset_index(drop=True)

    # Ensure matching matrix shapes
    min_len = min(tfidf_df.shape[0], emb_df.shape[0], corpus_embs.shape[0])
    tfidf_df, emb_df = tfidf_df.head(min_len), emb_df.head(min_len)
    tfidf_matrix = tfidf_matrix[:min_len]
    corpus_embs = corpus_embs[:min_len]
    print(f"✅ Aligned TF-IDF and Embedding datasets on {min_len} common papers.\n")
else:
    print("⚠️ Warning: Missing 'paper_id' column; assuming same order of rows.")
    min_len = min(tfidf_matrix.shape[0], corpus_embs.shape[0])
    tfidf_matrix = tfidf_matrix[:min_len]
    corpus_embs = corpus_embs[:min_len]
    tfidf_df = tfidf_df.head(min_len)
    emb_df = emb_df.head(min_len)

# 🧠 Load SentenceTransformer (E5-base)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/e5-base-v2", device=device)

print(f"✅ TF-IDF papers: {len(tfidf_df)} | E5 embeddings: {len(emb_df)} | Device: {device}\n")


# -------------------------------------------------------------------
# 🎯 Hybrid Recommendation Function
# -------------------------------------------------------------------
def hybrid_recommendation(query_input, alpha: float = 0.5, top_k: int = 5) -> pd.DataFrame:
    """
    Finds top-k reviewer authors using a hybrid similarity metric.

    Args:
        query_input (str | Path): Raw text or PDF path.
        alpha (float): Weight for TF-IDF similarity (0–1).
        top_k (int): Number of top reviewers to return.

    Returns:
        pd.DataFrame: DataFrame of authors with 'author_id' and 'similarity'.
    """
    # -------------------------------------------------------------
    # 1️⃣ Handle Input
    # -------------------------------------------------------------
    if isinstance(query_input, (str, Path)) and str(query_input).lower().endswith(".pdf"):
        print(f"📄 Extracting text from PDF: {query_input}")
        query_text = extract_text_from_pdf(query_input)
    else:
        query_text = str(query_input)

    query_text = clean_text(query_text)
    if len(query_text.strip()) < 50:
        raise ValueError("❌ Query text too short. Please use a full abstract or paper content.")

    # -------------------------------------------------------------
    # 2️⃣ TF-IDF Similarity
    # -------------------------------------------------------------
    q_tfidf = vectorizer.transform([query_text])
    sims_tfidf = cosine_similarity(q_tfidf, tfidf_matrix).flatten()

    # -------------------------------------------------------------
    # 3️⃣ Semantic (E5) Similarity
    # -------------------------------------------------------------
    q_emb = model.encode([f"query: {query_text}"], normalize_embeddings=True)
    sims_sem = cosine_similarity(q_emb, corpus_embs).flatten()

    # -------------------------------------------------------------
    # 4️⃣ Normalize and Combine
    # -------------------------------------------------------------
    eps = 1e-8
    sims_tfidf = (sims_tfidf - sims_tfidf.min()) / (sims_tfidf.max() - sims_tfidf.min() + eps)
    sims_sem = (sims_sem - sims_sem.min()) / (sims_sem.max() - sims_sem.min() + eps)

    hybrid_sims = alpha * sims_tfidf + (1 - alpha) * sims_sem

    if len(hybrid_sims) != len(emb_df):
        raise ValueError(
            f"❌ Hybrid mismatch — sims({len(hybrid_sims)}) vs emb_df({len(emb_df)})"
        )

    emb_df["hybrid_similarity"] = hybrid_sims

    # -------------------------------------------------------------
    # 5️⃣ Aggregate by Author (max for stronger topical relevance)
    # -------------------------------------------------------------
    ranked = (
        emb_df.groupby("author_id")["hybrid_similarity"]
        .max()
        .sort_values(ascending=False)
        .head(top_k)
    )

    results = pd.DataFrame({
        "author_id": ranked.index,
        "similarity": ranked.values
    }).reset_index(drop=True)

    # -------------------------------------------------------------
    # 6️⃣ Output Results
    # -------------------------------------------------------------
    print(f"\n🎯 Top {top_k} Reviewers (Hybrid α={alpha:.2f}):")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['author_id']:<25} | Score: {row['similarity']:.4f}")

    return results


# -------------------------------------------------------------------
# 🧪 CLI Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Reviewer Recommendation (TF-IDF + Semantic).")
    parser.add_argument("--file", type=str, help="Path to research paper (PDF).")
    parser.add_argument("--text", type=str, help="Raw text or abstract of the paper.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top reviewers to return.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for TF-IDF similarity.")
    args = parser.parse_args()

    query_input = args.file if args.file else args.text
    if not query_input:
        print("⚠️ Please provide either --file or --text argument.")
        exit(1)

    results = hybrid_recommendation(query_input, alpha=args.alpha, top_k=args.topk)
    print("\n✅ Hybrid reviewer recommendation complete.")
