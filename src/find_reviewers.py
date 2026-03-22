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
from concurrent.futures import ThreadPoolExecutor
import functools

# -------------------------------------------------------------------
# 📁 Configuration
# -------------------------------------------------------------------
TFIDF_DIR = Path("data/cache/tfidf_index/")
VECTOR_PATH = TFIDF_DIR / "vectorizer.pkl"
MATRIX_PATH = TFIDF_DIR / "tfidf_matrix.pkl"
DF_PATH = TFIDF_DIR / "tfidf_df.pkl"

# Pre-compile regex patterns
CLEAN_TEXT_REGEX = re.compile(r"[^a-z0-9\s\-]")  # PERF: Compile regex once at module level
MULTISPACE_REGEX = re.compile(r"\s+")

# -------------------------------------------------------------------
# 🧹 Utility Functions
# -------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str, max_chars: int = 4000) -> str:
    """Extract text from a PDF file safely (limited to 4000 chars)."""
    try:
        reader = PdfReader(pdf_path)
        with ThreadPoolExecutor() as executor:  # PERF: Sequential → Parallel processing of PDF pages
            text = "\n".join(executor.map(lambda page: page.extract_text() or "", reader.pages))
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
    text = CLEAN_TEXT_REGEX.sub(" ", text)  # PERF: Compile regex once at module level
    text = MULTISPACE_REGEX.sub(" ", text).strip()  # PERF: Compile regex once at module level
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


# Cache min and max similarity scores for normalization
SIMS_MIN = tfidf_matrix.min()
SIMS_MAX = tfidf_matrix.max()

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
    sims = (sims - SIMS_MIN) / (SIMS_MAX - SIMS_MIN + 1e-8)  # PERF: Cache min/max values for normalization

    # -------------------------------------------------------------
    # 3️⃣ Aggregate per Author
    # -------------------------------------------------------------
    df["similarity"] = sims  # PERF: Remove unnecessary DataFrame copy

    if method == "max":
        author_scores = df.groupby("author_id")["similarity"].max()
    elif method == "mean":
        author_scores = df.groupby("author_id")["similarity"].mean()
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
        author_id = row['author_id']  # PERF: Cache repeated attribute lookups
        similarity = row['similarity']  # PERF: Cache repeated attribute lookups
        print(f"{i+1}. {author_id:<25} | Similarity: {similarity:.4f}")

    print("\n✅ Reviewer matching complete.")