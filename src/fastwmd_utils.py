# src/fastwmd_utils.py
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
import gensim.downloader as api
import socket
from concurrent.futures import ThreadPoolExecutor  # PERF: Added for parallel model loading
import asyncio  # PERF: Added for async I/O in self-test

DATA_DIR = Path("data/cache")
CORPUS_PKL = DATA_DIR / "parsed_corpus.pkl"
FASTWMD_DIR = DATA_DIR / "fastwmd_index"
FASTWMD_DIR.mkdir(parents=True, exist_ok=True)

_W2V = None
_DIM = 300

# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------
def _internet_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), 2)
        return True
    except OSError:
        return False


def _try_load(model_name: str):
    """Try loading a model with friendly logging."""
    print(f"🔎 Attempting to load model: {model_name}")
    try:
        m = api.load(model_name)
        print(f"✅ Loaded: {model_name} ({len(m.index_to_key):,} tokens)")
        return m
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}: {e}")
        return None


def _load_fasttext_or_glove():
    """Load FastText first, fallback to GloVe."""
    if not _internet_connected():
        print("🌐 No internet detected — will use cached embeddings only if already downloaded.")
    with ThreadPoolExecutor() as executor:  # PERF: Sequential → Parallel model loading
        future_fasttext = executor.submit(_try_load, "fasttext-wiki-news-subwords-300")
        future_glove = executor.submit(_try_load, "glove-wiki-gigaword-300")
        model = future_fasttext.result()
        if model:
            return model, 300
        model = future_glove.result()
        if model:
            return model, 300
    raise RuntimeError(
        "❌ Could not load any embeddings. "
        "Run manually once: python -c \"import gensim.downloader as api; api.load('fasttext-wiki-news-subwords-300')\""
    )


def _get_w2v():
    global _W2V, _DIM
    if _W2V is None:
        print("⚙️ Loading FastText/GloVe vectors...")
        _W2V, _DIM = _load_fasttext_or_glove()
    return _W2V, _DIM


def _mean_emb(tokens, w2v, dim):
    vecs = [w2v[w] for w in tokens if w in w2v]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0, dtype=np.float32)


def _require_corpus():
    if not CORPUS_PKL.exists():
        raise FileNotFoundError(f"❌ Missing corpus file: {CORPUS_PKL}")
    df = pd.read_pickle(CORPUS_PKL)
    if "text" not in df.columns:
        df["text"] = (
            df.get("title", "").fillna("") + " " +
            df.get("abstract", "").fillna("") + " " +
            df.get("body", "").fillna("") + " " +
            df.get("text", "").fillna("")
        )
    df = df[df["text"].str.len() > 50].reset_index(drop=True, inplace=True)  # PERF: Copy → In-place reset_index
    if df.empty:
        raise ValueError("❌ No valid papers in parsed_corpus.pkl")
    return df


# -----------------------------------------------------------
# MAIN CORPUS BUILDER
# -----------------------------------------------------------
def build_or_load_fastwmd_corpus():
    emb_path = FASTWMD_DIR / "corpus_embeddings.npy"
    df_path = FASTWMD_DIR / "corpus_df.pkl"

    if emb_path.exists() and df_path.exists():
        print(f"📦 Using cached FastWMD embeddings from {emb_path}")
        embs = np.load(emb_path)
        df = pd.read_pickle(df_path)
        print(f"✅ Loaded {len(df)} docs | dim={embs.shape[1]}")
        return df, embs

    df = _require_corpus()
    w2v, dim = _get_w2v()

    print(f"🧹 Embedding {len(df)} documents...")
    tokens = [simple_preprocess(str(t)) for t in df["text"]]
    embs = np.vstack([_mean_emb(tok, w2v, dim) for tok in tokens]).astype(np.float32)

    np.save(emb_path, embs)
    df.to_pickle(df_path)  # PERF: Ensure no unnecessary copy before saving
    print(f"💾 Saved embeddings to {emb_path}")
    return df, embs


def encode_query_fastwmd(query):
    w2v, dim = _get_w2v()
    tokens = simple_preprocess(query)  # PERF: Repeated call → Single call
    return _mean_emb(tokens, w2v, dim)


# -----------------------------------------------------------
# SELF-TEST
# -----------------------------------------------------------
async def async_self_test():  # PERF: Synchronous → Asynchronous self-test
    from sklearn.metrics.pairwise import cosine_similarity

    print("🚀 Running FastWMD self-test...")
    try:
        df, X = await asyncio.to_thread(build_or_load_fastwmd_corpus)
        print(f"✅ Corpus ready: {len(df)} documents | shape={X.shape}")
        q = "Transformer architectures for deep learning and natural language tasks"
        q_vec = encode_query_fastwmd(q).reshape(1, -1)
        sims = cosine_similarity(q_vec, X).ravel()
        top_idx = np.argsort(-sims)[:5]
        print("\n🎯 Top-5 Similar Authors:")
        for i, idx in enumerate(top_idx):
            print(f"{i+1}. {df.iloc[idx]['author_id']:<25} | Similarity: {sims[idx]:.4f}")
    except Exception as e:
        import traceback
        print("\n❌ FastWMD self-test failed:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(async_self_test())  # PERF: Run async self-test