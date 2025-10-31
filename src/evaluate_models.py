"""
evaluate_models.py
==================
Final 2025-10 Edition — Hugging Face–safe

🧩 Evaluates all reviewer recommendation models:
   - TF-IDF
   - Semantic (E5)
   - Hybrid
   - Topic (LDA/NMF)

✅ Timestamped CSV exports (no overwriting)
✅ Independent pairwise model correlations
✅ Streamlit & Spaces compatible (Agg backend)
✅ Skips missing models gracefully
"""

from __future__ import annotations
import io
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

# -----------------------------------------------------------------------------
# 📁 Path Setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = PROJECT_ROOT.parent if (PROJECT_ROOT / "src").exists() else PROJECT_ROOT
DATA_DIR = APP_ROOT / "data"
TEST_PDF_DIR = APP_ROOT / "test_pdfs"
EVAL_DIR = DATA_DIR / "eval"

for d in [DATA_DIR, TEST_PDF_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# 🔧 Imports
# -----------------------------------------------------------------------------
from src.find_reviewers import find_top_reviewers as tfidf_model
from src.find_reviewers_semantic import find_top_reviewers as semantic_model
from src.find_reviewers_hybrid import hybrid_recommendation
from src.find_reviewers_topic import find_top_reviewers_topic  # ✅ Topic model included

# -----------------------------------------------------------------------------
# 🧹 Utilities
# -----------------------------------------------------------------------------
def extract_text_from_pdf(path: Path, max_chars: int = 4000) -> str:
    """Extract readable text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️ Error reading PDF {path.name}: {e}")
        return ""

def _normalize_similarity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure uniform column naming for similarity."""
    df = df.copy()
    if "similarity" not in df.columns:
        if "score" in df.columns:
            df.rename(columns={"score": "similarity"}, inplace=True)
        elif "hybrid_similarity" in df.columns:
            df.rename(columns={"hybrid_similarity": "similarity"}, inplace=True)
    return df

# -----------------------------------------------------------------------------
# 📊 Visualization
# -----------------------------------------------------------------------------
def fig_similarity_hist(tfidf_df, sem_df, hyb_df, topic_df):
    fig = plt.figure(figsize=(6, 4))
    for df, label in [
        (tfidf_df, "TF-IDF"),
        (sem_df, "Semantic (E5)"),
        (hyb_df, "Hybrid"),
        (topic_df, "Topic (LDA/NMF)"),
    ]:
        if not df.empty and "similarity" in df.columns:
            plt.hist(df["similarity"].dropna(), bins=20, alpha=0.5, density=True, label=label)
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Model Similarity Distributions")
    plt.tight_layout()
    return fig

def fig_corr_heatmap(corr_df):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_df.values, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
    plt.title("Model-to-Model Correlation Matrix")
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# 🧩 Batch Evaluation
# -----------------------------------------------------------------------------
def export_results(pdf_dir: Path, top_k: int = 5):
    """Run all models on PDFs and export timestamped CSVs."""
    pdf_dir = Path(pdf_dir).expanduser().resolve()
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    print(f"📂 Found {len(pdf_files)} PDFs in {pdf_dir}")
    if not pdf_files:
        raise FileNotFoundError("❌ No PDFs found in folder.")

    all_tfidf, all_sem, all_hyb, all_topic = [], [], [], []

    for pdf in pdf_files:
        print(f"\n📄 Processing {pdf.name}...")
        query_text = extract_text_from_pdf(pdf)
        if not query_text.strip():
            print(f"⚠️ Skipping {pdf.name} — no extractable text.")
            continue

        try:
            res_tfidf = _normalize_similarity_columns(tfidf_model(query_text, top_k=top_k))
            res_sem   = _normalize_similarity_columns(semantic_model(query_text, top_k=top_k))
            res_hyb   = _normalize_similarity_columns(hybrid_recommendation(query_text, alpha=0.4, top_k=top_k))
            res_topic = _normalize_similarity_columns(find_top_reviewers_topic(query_text, top_k=top_k, model="lda"))

            for df, store in [
                (res_tfidf, all_tfidf),
                (res_sem, all_sem),
                (res_hyb, all_hyb),
                (res_topic, all_topic),
            ]:
                if not df.empty:
                    df["query_id"] = pdf.stem
                    store.append(df)

        except Exception as e:
            print(f"❌ Error evaluating {pdf.name}: {e}")
            continue

    # Combine and export
    def concat_or_empty(lst):
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()

    tfidf_df, sem_df, hyb_df, topic_df = map(concat_or_empty, [all_tfidf, all_sem, all_hyb, all_topic])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, df in [
        ("tfidf", tfidf_df),
        ("semantic", sem_df),
        ("hybrid", hyb_df),
        ("topic", topic_df),
    ]:
        if not df.empty:
            out_path = EVAL_DIR / f"{name}_results_{timestamp}.csv"
            df.to_csv(out_path, index=False)
            print(f"💾 Saved {name} results → {out_path}")

    return tfidf_df, sem_df, hyb_df, topic_df

# -----------------------------------------------------------------------------
# 📈 Correlation Computation
# -----------------------------------------------------------------------------
def model_correlation(df1: pd.DataFrame, df2: pd.DataFrame):
    df1 = _normalize_similarity_columns(df1)
    df2 = _normalize_similarity_columns(df2)
    if df1.empty or df2.empty:
        return np.nan, np.nan
    merged = df1.merge(df2, on=["query_id", "author_id"], suffixes=("_1", "_2"))
    if merged.empty:
        return np.nan, np.nan
    pearson = pearsonr(merged["similarity_1"], merged["similarity_2"])[0]
    spearman = spearmanr(merged["similarity_1"], merged["similarity_2"])[0]
    return float(pearson), float(spearman)

# -----------------------------------------------------------------------------
# 🧪 Evaluation Runner
# -----------------------------------------------------------------------------
def run_full_evaluation(
    pdf_dir: Path | str = TEST_PDF_DIR,
    k: int = 5,
    top_k_export: int = 5,
    return_results: bool = True,
    show_plots: bool = False
):
    pdf_dir = Path(pdf_dir).expanduser().resolve()
    tfidf_df, sem_df, hyb_df, topic_df = export_results(pdf_dir, top_k=top_k_export)

    print("\n🔗 Computing pairwise model correlations...\n")

    models = {
        "TF-IDF": tfidf_df,
        "Semantic (E5)": sem_df,
        "Hybrid": hyb_df,
        "Topic (LDA/NMF)": topic_df,
    }

    corr_matrix = pd.DataFrame(index=models.keys(), columns=models.keys(), dtype=float)

    for name1, df1 in models.items():
        for name2, df2 in models.items():
            pearson, _ = model_correlation(df1, df2)
            corr_matrix.loc[name1, name2] = pearson

    figs = {
        "similarity_hist": fig_similarity_hist(tfidf_df, sem_df, hyb_df, topic_df),
        "corr_heatmap": fig_corr_heatmap(corr_matrix),
    }

    print("\n📈 Model-to-Model Correlation Matrix:")
    print(corr_matrix.round(3))

    if return_results:
        return tfidf_df, sem_df, hyb_df, topic_df, corr_matrix, figs

# -----------------------------------------------------------------------------
# 🚀 CLI Mode
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, default=str(TEST_PDF_DIR), help="Folder containing test PDFs")
    ap.add_argument("--topk", type=int, default=5, help="Top-K reviewers per model")
    args = ap.parse_args()
    run_full_evaluation(pdf_dir=args.pdf_dir, top_k_export=args.topk)
