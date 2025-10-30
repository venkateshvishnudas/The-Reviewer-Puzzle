"""
build_tfidf_index.py
====================

Creates a TF-IDF index for all parsed research papers in the corpus.
This index will allow you to find semantically similar papers
and identify top-k potential reviewers.

Usage:
------
python src/build_tfidf_index.py

Later, you can import the helper function:
from src.build_tfidf_index import find_similar_papers
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from tqdm import tqdm


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
CORPUS_PATH = Path("data/cache/parsed_corpus.pkl")
OUTPUT_PATH = Path("data/cache/tfidf_index/")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# 1️⃣ Load Corpus
# -------------------------------------------------------------------
print("📘 Loading parsed corpus...")
df = pd.read_pickle(CORPUS_PATH)
print(f"✅ Loaded {len(df)} papers from corpus.")


# -------------------------------------------------------------------
# 2️⃣ Preprocess Text
# -------------------------------------------------------------------
def clean_text(text):
    """Basic text cleaner to normalize whitespace and remove unwanted chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print("🧹 Cleaning text content...")
df["combined_text"] = (
    df["title"].fillna("") + " " +
    df["abstract"].fillna("") + " " +
    df["body"].fillna("") + " " +
    df["text"].fillna("")
)
df["combined_text"] = df["combined_text"].apply(clean_text)

# drop empty docs
df = df[df["combined_text"].str.len() > 50].reset_index(drop=True)
print(f"✅ {len(df)} papers retained after cleaning.")


# -------------------------------------------------------------------
# 3️⃣ Build TF-IDF Index
# -------------------------------------------------------------------
print("⚙️ Building TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

tfidf_matrix = vectorizer.fit_transform(tqdm(df["combined_text"], desc="Vectorizing"))

print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")


# -------------------------------------------------------------------
# 4️⃣ Save Model + Data
# -------------------------------------------------------------------
print("💾 Saving vectorizer and matrix...")
joblib.dump(vectorizer, OUTPUT_PATH / "vectorizer.pkl")
joblib.dump(tfidf_matrix, OUTPUT_PATH / "tfidf_matrix.pkl")
df.to_pickle(OUTPUT_PATH / "tfidf_df.pkl")
print(f"✅ TF-IDF index built for {len(df)} papers.")
print(f"📁 Saved to: {OUTPUT_PATH.resolve()}")


# -------------------------------------------------------------------
# 5️⃣ Query Function
# -------------------------------------------------------------------
def find_similar_papers(query_text: str, top_k: int = 5):
    """
    Given a query text (e.g., a new research paper),
    find top-k most similar papers and their authors.
    """
    # Lazy-load cached models if not loaded already
    global vectorizer, tfidf_matrix, df
    if "vectorizer" not in globals():
        vectorizer = joblib.load(OUTPUT_PATH / "vectorizer.pkl")
        tfidf_matrix = joblib.load(OUTPUT_PATH / "tfidf_matrix.pkl")
        df = pd.read_pickle(OUTPUT_PATH / "tfidf_df.pkl")

    cleaned_query = clean_text(query_text)
    query_vec = vectorizer.transform([cleaned_query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = sims.argsort()[::-1][:top_k]
    results = df.iloc[top_indices][["author_id", "title", "paper_id"]].copy()
    results["similarity_score"] = sims[top_indices]
    return results.reset_index(drop=True)


# -------------------------------------------------------------------
# 6️⃣ (Optional) Interactive Test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔍 Example Query:")
    sample_query = "Deep learning approaches for image recognition in healthcare"
    print(find_similar_papers(sample_query, top_k=5))
