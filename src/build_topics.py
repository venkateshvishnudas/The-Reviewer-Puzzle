"""
build_topics.py
===============

Applies LDA and NMF topic modeling on parsed corpus and saves topic distributions.
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import numpy as np
import re, joblib

DATA_DIR = Path("data/cache")
CORPUS_PKL = DATA_DIR / "parsed_corpus.pkl"
SAVE_DIR = DATA_DIR / "topics"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(s):
    if not isinstance(s, str): return ""
    s = re.sub(r"[^a-zA-Z\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s)
    return s.strip()

print("📦 Loading corpus...")
df = pd.read_pickle(CORPUS_PKL)
df["text"] = (
    df["title"].fillna("") + " " +
    df["abstract"].fillna("") + " " +
    df["body"].fillna("") + " " +
    df["text"].fillna("")
)
df["clean_text"] = df["text"].apply(clean_text)

print("⚙️ Building Vectorizers...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
count = CountVectorizer(max_features=5000, stop_words="english")

X_tfidf = tfidf.fit_transform(df["clean_text"])
X_count = count.fit_transform(df["clean_text"])

print("🧠 Training LDA...")
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda_topics = lda.fit_transform(X_count)

print("🧠 Training NMF...")
nmf = NMF(n_components=10, random_state=42)
nmf_topics = nmf.fit_transform(X_tfidf)

print("💾 Saving topic models and vectorizers...")
np.save(SAVE_DIR / "lda_topics.npy", lda_topics)
np.save(SAVE_DIR / "nmf_topics.npy", nmf_topics)
df.to_pickle(SAVE_DIR / "topics_df.pkl")

joblib.dump(lda, SAVE_DIR / "lda_model.pkl")
joblib.dump(nmf, SAVE_DIR / "nmf_model.pkl")
joblib.dump(count, SAVE_DIR / "count_vectorizer.pkl")
joblib.dump(tfidf, SAVE_DIR / "tfidf_vectorizer.pkl")

print(f"✅ Saved LDA & NMF topic models for {len(df)} papers.")
