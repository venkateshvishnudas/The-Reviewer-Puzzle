"""
find_reviewers.py (v2.0 strict)
===============================

TF-IDF-based Reviewer Recommendation System
--------------------------------------------
Uses a pre-built TF-IDF index to recommend the most relevant reviewers.

Key Fixes:
- Ensures alignment between TF-IDF matrix and metadata
- Prevents global df contamination
- Safer PDF extraction
- Retains domain tokens (numbers, hyphens)
- Default aggregation: max (strongest topical match)
- Normalized similarity scaling
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import re
import argparse
import sys, os

# -------------------------------------------------------------------
# 📁 Configuration
# -------------------------------------------------------------------
TFIDF_DIR = Path("data/cache/tfidf_index/")
VECTOR_PATH = TFIDF_DIR / "vectorizer.pkl"
MATRIX_PATH = TFIDF_DIR / "tfidf_matrix.pkl"
DF_PATH = TFIDF_DIR / "tfidf_df.pkl"

# -------------------------------------------------------------------
# 🧹 Utility Functions
# -------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str, max_chars: int = 4000) -> str:
    """Extract text from a PDF file safely (limited to 4000 chars)."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text[:max_chars]
    except PdfReadError:
        print(f"⚠️ Could not fully read PDF: {pdf_path}")
        return ""
    except Exception as e:
        print(f"⚠️ PDF read error: {e}")
        return ""


def clean_text(text: str) -> str:
    """Normalize text while retaining digits and hyphens."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -------------------------------------------------------------------
# ⚙️ Load TF-IDF Models
# -------------------------------------------------------------------
print("📦 Loading TF-IDF index...")
vectorizer = joblib.load(VECTOR_PATH)
tfidf_matrix = joblib.load(MATRIX_PATH)
df = pd.read_pickle(DF_PATH)

if tfidf_matrix.shape[0] != len(df):
    raise ValueError(
        f"❌ Mismatch between TF-IDF matrix rows ({tfidf_matrix.shape[0]}) and df entries ({len(df)})."
    )

print(f"✅ Loaded {len(df)} papers from {df['author_id'].nunique()} authors.\n")


# -------------------------------------------------------------------
# 🎯 Core Reviewer Finder
# -------------------------------------------------------------------
def find_top_reviewers(query_input, top_k: int = 5, method: str = "max") -> pd.DataFrame:
    """
    Find top-k authors suitable to review a paper via TF-IDF similarity.

    Args:
        query_input (str | Path): Text or PDF file path.
        top_k (int): Number of reviewers to recommend.
        method (str): Aggregation method — 'max' or 'mean'.

    Returns:
        pd.DataFrame: Ranked authors with similarity scores.
    """
    # -------------------------------------------------------------
    # 1️⃣ Handle Input
    # -------------------------------------------------------------
    if isinstance(query_input, (str, Path)) and str(query_input).lower().endswith(".pdf"):
        print(f"📄 Extracting text from PDF: {query_input}")
        query_text = extract_text_from_pdf(query_input)
    else:
        query_text = str(query_input)

    if len(query_text.strip()) < 50:
        raise ValueError("❌ Query text too short. Provide a full abstract or content.")

    # -------------------------------------------------------------
    # 2️⃣ Vectorize and Compute Similarity
    # -------------------------------------------------------------
    cleaned_query = clean_text(query_text)
    query_vec = vectorizer.transform([cleaned_query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Normalize for consistency
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # -------------------------------------------------------------
    # 3️⃣ Aggregate per Author
    # -------------------------------------------------------------
    df_local = df.copy()
    df_local["similarity"] = sims

    if method == "max":
        author_scores = df_local.groupby("author_id")["similarity"].max()
    elif method == "mean":
        author_scores = df_local.groupby("author_id")["similarity"].mean()
    else:
        raise ValueError("⚠️ method must be 'max' or 'mean'")

    ranked = author_scores.sort_values(ascending=False).head(top_k)

    results = pd.DataFrame({
        "author_id": ranked.index,
        "similarity": ranked.values
    }).reset_index(drop=True)

    return results


# -------------------------------------------------------------------
# 🧪 CLI Example Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top reviewers using TF-IDF similarity.")
    parser.add_argument("--file", type=str, help="Path to research paper (PDF).")
    parser.add_argument("--text", type=str, help="Raw text or abstract of the paper.")
    parser.add_argument("--topk", type=int, default=5, help="Number of reviewers to recommend.")
    parser.add_argument("--method", type=str, default="max", choices=["max", "mean"],
                        help="Aggregation method per author.")
    args = parser.parse_args()

    if not args.file and not args.text:
        print("⚠️ Please provide either --file or --text argument.")
        sys.exit(1)

    query_input = args.file if args.file else args.text

    results = find_top_reviewers(query_input, top_k=args.topk, method=args.method)

    print("\n🎯 Top Potential Reviewers:")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['author_id']:<25} | Similarity: {row['similarity']:.4f}")

    print("\n✅ Reviewer matching complete.")
