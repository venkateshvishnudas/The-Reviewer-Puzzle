"""
find_reviewers_meanemb.py
=========================

Reviewer Recommendation using Mean Word Embeddings
--------------------------------------------------

This model computes document and query embeddings by averaging word vectors
from pretrained FastText or GloVe embeddings (loaded via Gensim).

It serves as an efficient, interpretable baseline for semantic similarity,
approximating Word Mover’s Distance (WMD) without the computational cost.

Supports:
- PDF and raw text input
- Reusable cached corpus embeddings
- Streamlit and CLI integration

Usage:
------
python src/find_reviewers_meanemb.py --file path/to/paper.pdf --topk 5
"""

from pathlib import Path
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re

# -------------------------------------------------------------------
# 📁 Configuration
# -------------------------------------------------------------------
DATA_DIR = Path("data/cache")
CORPUS_PKL = DATA_DIR / "parsed_corpus.pkl"
MEANEMB_DIR = DATA_DIR / "mean_emb_index"
MEANEMB_DIR.mkdir(parents=True, exist_ok=True)

_W2V, _DIM = None, 300


# -------------------------------------------------------------------
# 🧠 Model Loader
# -------------------------------------------------------------------
def _load_model():
    """Load pretrained FastText or fallback to GloVe embeddings."""
    global _W2V, _DIM
    if _W2V is not None:
        return _W2V, _DIM

    try:
        print("⚙️ Loading FastText subword vectors (300d)...")
        _W2V = api.load("fasttext-wiki-news-subwords-300")
    except Exception as e:
        print(f"⚠️ FastText load failed ({e}). Falling back to GloVe...")
        _W2V = api.load("glove-wiki-gigaword-300")

    _DIM = _W2V.vector_size
    print(f"✅ Loaded embeddings ({_DIM}-dimensional, {len(_W2V.index_to_key):,} tokens).")
    return _W2V, _DIM


# -------------------------------------------------------------------
# 🧹 Utilities
# -------------------------------------------------------------------
def _mean_emb(tokens, w2v):
    """Compute mean embedding vector for a tokenized text."""
    vecs = [w2v[w] for w in tokens if w in w2v]
    if not vecs:
        return np.zeros(w2v.vector_size, dtype=np.float32)
    vec = np.mean(vecs, axis=0).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def extract_text_from_pdf(pdf_path: Path, max_chars: int = 4000) -> str:
    """Extract clean text from a PDF (first few thousand characters)."""
    try:
        reader = PdfReader(str(pdf_path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️ Could not extract text from {pdf_path.name}: {e}")
        return ""


def clean_text(s: str) -> str:
    """Basic text cleaning for preprocessing."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[^a-zA-Z\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------------------------------------------------
# 🧩 Build or Load Corpus Embeddings
# -------------------------------------------------------------------
def build_or_load_corpus():
    """Build (or load cached) mean embeddings for all corpus papers."""
    emb_path = MEANEMB_DIR / "corpus_embeddings.npy"
    df_path = MEANEMB_DIR / "corpus_df.pkl"

    if emb_path.exists() and df_path.exists():
        df = pd.read_pickle(df_path)
        X = np.load(emb_path)
        print(f"📦 Loaded {len(df)} cached documents ({X.shape[1]}-dimensional).")
        return df, X

    if not CORPUS_PKL.exists():
        raise FileNotFoundError(f"❌ Missing corpus file: {CORPUS_PKL}. Run parsing first.")

    print("📘 Loading and preprocessing corpus...")
    df = pd.read_pickle(CORPUS_PKL)
    df["text"] = (
        df["title"].fillna("") + " " +
        df["abstract"].fillna("") + " " +
        df.get("body", "").fillna("") + " " +
        df.get("text", "").fillna("")
    )
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    w2v, _ = _load_model()
    tokens = [simple_preprocess(t) for t in df["text"]]
    print(f"🧠 Generating mean embeddings for {len(tokens)} documents...")
    X = np.vstack([_mean_emb(tok, w2v) for tok in tokens])
    np.save(emb_path, X)
    df.to_pickle(df_path)

    print(f"💾 Saved embeddings → {emb_path.name}")
    return df, X


# -------------------------------------------------------------------
# 🔍 Query Encoding
# -------------------------------------------------------------------
def encode_query(query_input):
    """Encode a query (text or PDF) into a mean embedding vector."""
    if isinstance(query_input, (str, Path)) and str(query_input).lower().endswith(".pdf"):
        query_text = extract_text_from_pdf(query_input)
    else:
        query_text = str(query_input)

    query_text = clean_text(query_text)
    w2v, _ = _load_model()
    tokens = simple_preprocess(query_text)
    return _mean_emb(tokens, w2v)


# -------------------------------------------------------------------
# 🎯 Main Reviewer Finder
# -------------------------------------------------------------------
def find_top_reviewers(query_input, top_k=5):
    """
    Recommend top-K reviewers using Mean Word Embedding similarity.

    Args:
        query_input (str | Path): Text or path to research paper PDF.
        top_k (int): Number of reviewers to recommend.

    Returns:
        pd.DataFrame: Ranked reviewers with similarity scores.
    """
    df, X = build_or_load_corpus()
    q_vec = encode_query(query_input).reshape(1, -1)

    sims = cosine_similarity(q_vec, X).ravel()
    df["similarity"] = sims

    ranked = (
        df.groupby("author_id")["similarity"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
    )

    results = pd.DataFrame({
        "author_id": ranked.index,
        "similarity": ranked.values
    }).reset_index(drop=True)

    print(f"\n🎯 Top {top_k} Reviewers (Mean Embedding Cosine Similarity):")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['author_id']:<25} | {row['similarity']:.4f}")

    return results


# -------------------------------------------------------------------
# 🧪 CLI Example Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reviewer recommendation using mean word embeddings.")
    parser.add_argument("--file", type=str, help="Path to research paper PDF.")
    parser.add_argument("--text", type=str, help="Raw text or abstract.")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    query_input = args.file if args.file else args.text
    if not query_input:
        print("⚠️ Please provide either --file or --text argument.")
        exit(1)

    results = find_top_reviewers(query_input, top_k=args.topk)
    print("\n✅ Returned Results DataFrame:")
    print(results)
