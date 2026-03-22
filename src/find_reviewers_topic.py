"""
find_reviewers_topic.py
========================

Strict and corrected version (2025-10)

Topic Modeling–based Reviewer Recommendation System.

- Supports LDA or NMF topic modeling
- Uses pre-fitted vectorizers + models
- Computes cosine similarity between topic vectors
- Aggregates by author (max-based for stronger topical matching)
- Avoids re-transforming corpus every run
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
import joblib
from PyPDF2 import PdfReader
from functools import lru_cache
from collections import defaultdict

# -------------------------------------------------------------------
# 📁 Paths
# -------------------------------------------------------------------
DATA_DIR = Path("data/cache/topics")
LDA_MODEL_PATH = DATA_DIR / "lda_model.pkl"
NMF_MODEL_PATH = DATA_DIR / "nmf_model.pkl"
COUNT_VEC_PATH = DATA_DIR / "count_vectorizer.pkl"
TFIDF_VEC_PATH = DATA_DIR / "tfidf_vectorizer.pkl"
DF_PATH = DATA_DIR / "topics_df.pkl"

# Cache the loaded DataFrame to avoid repeated I/O operations
_cached_df = None

# -------------------------------------------------------------------
# 🧹 Text Cleaning
# -------------------------------------------------------------------
@lru_cache(maxsize=None)
def clean_text(s: str) -> str:
    """Clean text with domain-safe normalization for topic modeling."""
    if not isinstance(s, str):
        return ""
    # Keep alphanumerics, hyphens, and numbers
    s = re.sub(r"[^a-zA-Z0-9\s\-]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_text_from_pdf(pdf_path: Path, max_chars: int = 4000) -> str:
    """Extract readable text from a PDF."""
    try:
        reader = PdfReader(str(pdf_path))
        # PERF: String concatenation in loop → ''.join(parts) — more efficient memory usage
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return clean_text(text[:max_chars])
    except Exception as e:
        print(f"⚠️ Could not extract text from PDF: {e}")
        return ""

def safe_load(path: Path, desc: str):
    """Safely load serialized models."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Missing {desc} at: {path}")
    return joblib.load(path)

# -------------------------------------------------------------------
# 🎯 Main Function — Find Top Reviewers (LDA/NMF)
# -------------------------------------------------------------------
def find_top_reviewers_topic(
    query_input,
    top_k: int = 5,
    model: str = "lda",
) -> pd.DataFrame:
    """
    Compute topic similarity between query and author corpus using LDA or NMF.

    Args:
        query_input (str | Path): Raw text or PDF file path.
        top_k (int): Number of top reviewers to return.
        model (str): 'lda' or 'nmf'.

    Returns:
        pd.DataFrame: ['author_id', 'similarity'] for top matches.
    """
    global _cached_df
    print(f"\n📦 Loading topic models and vectorizers for {model.upper()}...\n")

    # Load base dataset
    if _cached_df is None:
        if not DF_PATH.exists():
            raise FileNotFoundError(f"❌ topics_df.pkl not found at {DF_PATH}")
        _cached_df = pd.read_pickle(DF_PATH)  # PERF: Repeated loading → Cached in memory

    df = _cached_df

    # Load model + vectorizer
    if model.lower() == "lda":
        topic_model = safe_load(LDA_MODEL_PATH, "LDA Model")
        vectorizer = safe_load(COUNT_VEC_PATH, "Count Vectorizer")
    elif model.lower() == "nmf":
        topic_model = safe_load(NMF_MODEL_PATH, "NMF Model")
        vectorizer = safe_load(TFIDF_VEC_PATH, "TF-IDF Vectorizer")
    else:
        raise ValueError("❌ Invalid model type — choose 'lda' or 'nmf'.")

    print(f"✅ Loaded {len(df)} papers from {df['author_id'].nunique()} authors.\n")

    # -------------------------------------------------------------
    # 1️⃣ Handle Input (PDF or text)
    # -------------------------------------------------------------
    if isinstance(query_input, (str, Path)) and str(query_input).lower().endswith(".pdf"):
        print(f"📄 Extracting text from PDF: {query_input}")
        query_text = extract_text_from_pdf(query_input)
    else:
        query_text = str(query_input)

    query_clean = clean_text(query_text)
    if len(query_clean) < 80:
        raise ValueError("⚠️ Query text too short for meaningful topic comparison.")

    # -------------------------------------------------------------
    # 2️⃣ Encode Query into Topic Space
    # -------------------------------------------------------------
    q_vec = vectorizer.transform([query_clean])
    q_topic = topic_model.transform(q_vec)
    q_topic = normalize(q_topic, norm="l2")

    # -------------------------------------------------------------
    # 3️⃣ Corpus Topic Representation (cached or computed)
    # -------------------------------------------------------------
    if "topic_vector" in df.columns:
        doc_topics = np.vstack(df["topic_vector"].values)
    else:
        print("⚙️ Computing topic vectors for corpus (first-time run)...")
        df["clean_text"] = df["text"].apply(clean_text)
        doc_vecs = vectorizer.transform(df["clean_text"])
        doc_topics = topic_model.transform(doc_vecs)
        df["topic_vector"] = list(doc_topics)
        df.to_pickle(DF_PATH)  # cache for next run

    doc_topics = normalize(doc_topics, norm="l2")

    # -------------------------------------------------------------
    # 4️⃣ Compute Cosine Similarity
    # -------------------------------------------------------------
    sims = cosine_similarity(q_topic, doc_topics).flatten()  # PERF: Consider using Faiss for large-scale similarity computations
    df["similarity"] = sims

    # -------------------------------------------------------------
    # 5️⃣ Aggregate by Author (use max, not mean)
    # -------------------------------------------------------------
    ranked = (
        df.groupby("author_id")["similarity"]
        .max()
        .sort_values(ascending=False)
        .head(top_k)
    )

    results = pd.DataFrame({
        "author_id": ranked.index,
        "similarity": ranked.values
    }).reset_index(drop=True)

    # -------------------------------------------------------------
    # 6️⃣ Display Results
    # -------------------------------------------------------------
    print(f"\n🎯 Top {top_k} Reviewers ({model.upper()} Topic Model):")
    for i, row in results.iterrows():
        author_id = row['author_id']  # PERF: Repeated attribute lookup → Cached in local variable
        print(f"{i+1}. {author_id:<25} | Similarity: {row['similarity']:.4f}")

    return results

# -------------------------------------------------------------------
# 🧪 CLI Example Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find top reviewers using LDA/NMF topic modeling.")
    parser.add_argument("--file", type=str, help="Path to research paper (PDF).")
    parser.add_argument("--text", type=str, help="Raw text or abstract of the paper.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model", type=str, default="lda", choices=["lda", "nmf"])
    args = parser.parse_args()

    query_input = args.file if args.file else args.text
    if not query_input:
        print("⚠️ Please provide either --file or --text argument.")
        exit(1)

    results = find_top_reviewers_topic(query_input, top_k=args.topk, model=args.model)
    print("\n✅ Returned Results DataFrame:")
    print(results)