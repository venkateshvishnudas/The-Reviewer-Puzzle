"""
evaluate_models.py
==================
Strict version (2025-10)

Dynamic evaluation of reviewer recommendation models.

✅ No dependency on manual labels
✅ No overwriting of CSVs (timestamped exports)
✅ True pairwise model correlation (no synthetic averages)
✅ Clean, Streamlit-compatible interface
✅ Headless-safe plotting (Agg backend)
"""

from __future__ import annotations
import io
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------
# Import models
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.find_reviewers import find_top_reviewers as tfidf_model
from src.find_reviewers_semantic import find_top_reviewers as semantic_model
from src.find_reviewers_hybrid import hybrid_recommendation

# -------------------------------
# Paths
# -------------------------------
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "test_pdfs"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Utility functions
# -------------------------------
def _normalize_author(s: str) -> str:
    """Normalize author names for fuzzy matching."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    for bad in ["dr.", "dr", "prof.", "prof", "ph.d", "phd", "mr.", "mrs.", "ms."]:
        s = s.replace(bad, " ")
    s = s.replace(".", " ").replace(",", " ").replace("_", " ")
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    s = " ".join(s.split())
    return s.strip()


def extract_text_from_pdf(path: Path | io.BytesIO, max_chars: int = 4000) -> str:
    """Extract text from a PDF file (either file path or bytes)."""
    try:
        reader = PdfReader(path if isinstance(path, (str, Path)) else io.BytesIO(path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text[:max_chars]
    except Exception as e:
        print(f"⚠️ Error reading PDF: {e}")
        return ""


def _normalize_similarity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a consistent 'similarity' column exists in model outputs."""
    df = df.copy()
    if "similarity" not in df.columns:
        if "score" in df.columns:
            df.rename(columns={"score": "similarity"}, inplace=True)
        elif "hybrid_similarity" in df.columns:
            df.rename(columns={"hybrid_similarity": "similarity"}, inplace=True)
    return df


# -------------------------------
# Plotting
# -------------------------------
def fig_similarity_hist(tfidf_df, sem_df, hyb_df):
    """Histogram of similarity score distributions."""
    fig = plt.figure(figsize=(6, 4))
    plt.hist(tfidf_df["similarity"].dropna(), bins=20, alpha=0.5, density=True, label="TF-IDF")
    plt.hist(sem_df["similarity"].dropna(), bins=20, alpha=0.5, density=True, label="Semantic (E5)")
    plt.hist(hyb_df["similarity"].dropna(), bins=20, alpha=0.5, density=True, label="Hybrid")
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Model Similarity Distribution")
    plt.tight_layout()
    return fig


def fig_corr_heatmap(corr_df):
    """Visualize correlation matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_df.values, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_df.index)
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    plt.title("Model-to-Model Correlation")
    plt.tight_layout()
    return fig


# -------------------------------
# Batch Evaluation (no manual labels)
# -------------------------------
def export_results(pdf_dir: Path, top_k: int = 5):
    """Run all models on PDFs in a folder and save CSV outputs (timestamped)."""
    pdf_files = sorted(list(pdf_dir.glob("*.pdf")))
    print(f"📂 Found {len(pdf_files)} PDFs for evaluation in {pdf_dir}")
    if not pdf_files:
        raise FileNotFoundError("No PDFs found in folder.")

    all_tfidf, all_sem, all_hyb = [], [], []

    for pdf in pdf_files:
        print(f"\n📄 Processing {pdf.name}...")
        query_text = extract_text_from_pdf(pdf)
        if not query_text.strip():
            print(f"⚠️ Skipping {pdf.name} — no extractable text.")
            continue

        try:
            res_tfidf = _normalize_similarity_columns(tfidf_model(query_text, top_k=top_k))
            res_sem = _normalize_similarity_columns(semantic_model(query_text, top_k=top_k))
            res_hyb = _normalize_similarity_columns(hybrid_recommendation(query_text, alpha=0.4, top_k=top_k))

            for df, lst in [(res_tfidf, all_tfidf), (res_sem, all_sem), (res_hyb, all_hyb)]:
                df["query_id"] = pdf.stem
                lst.append(df)

        except Exception as e:
            print(f"❌ Error evaluating {pdf.name}: {e}")
            continue

    if not all_tfidf or not all_sem or not all_hyb:
        raise RuntimeError("❌ No valid results generated from models.")

    tfidf_df = pd.concat(all_tfidf, ignore_index=True)
    sem_df = pd.concat(all_sem, ignore_index=True)
    hyb_df = pd.concat(all_hyb, ignore_index=True)

    # Timestamped exports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tfidf_path = EVAL_DIR / f"tfidf_results_{timestamp}.csv"
    sem_path = EVAL_DIR / f"semantic_results_{timestamp}.csv"
    hyb_path = EVAL_DIR / f"hybrid_results_{timestamp}.csv"

    tfidf_df.to_csv(tfidf_path, index=False)
    sem_df.to_csv(sem_path, index=False)
    hyb_df.to_csv(hyb_path, index=False)

    print(f"✅ Exported CSVs → {EVAL_DIR} (timestamped)")
    return tfidf_df, sem_df, hyb_df


# -------------------------------
# Correlation Helper
# -------------------------------
def model_correlation(df1: pd.DataFrame, df2: pd.DataFrame):
    """Compute Pearson & Spearman correlation between models."""
    df1 = _normalize_similarity_columns(df1)
    df2 = _normalize_similarity_columns(df2)
    merged = df1.merge(df2, on=["query_id", "author_id"], suffixes=("_1", "_2"))
    if merged.empty:
        return np.nan, np.nan
    pearson = pearsonr(merged["similarity_1"], merged["similarity_2"])[0]
    spearman = spearmanr(merged["similarity_1"], merged["similarity_2"])[0]
    return float(pearson), float(spearman)


# -------------------------------
# Evaluation Runner
# -------------------------------
def run_full_evaluation(
    pdf_dir: Path | str = DEFAULT_PDF_DIR,
    k: int = 5,
    top_k_export: int = 5,
    return_results: bool = True,
    show_plots: bool = False
):
    """Evaluate all models and compute independent pairwise correlations."""
    pdf_dir = Path(pdf_dir)
    tfidf_df, sem_df, hyb_df = export_results(pdf_dir, top_k=top_k_export)

    # Independent pairwise correlation computations
    pearson_tf_hyb, _ = model_correlation(tfidf_df, hyb_df)
    pearson_sem_hyb, _ = model_correlation(sem_df, hyb_df)
    pearson_tf_sem, _ = model_correlation(tfidf_df, sem_df)

    corr_df = pd.DataFrame(
        [
            [1.0, pearson_tf_hyb, pearson_tf_sem],
            [pearson_tf_hyb, 1.0, pearson_sem_hyb],
            [pearson_tf_sem, pearson_sem_hyb, 1.0],
        ],
        index=["TF-IDF", "Hybrid", "Semantic (E5)"],
        columns=["TF-IDF", "Hybrid", "Semantic (E5)"]
    )

    print("\n🔗 Model Correlation Matrix:")
    print(corr_df.round(3))

    figs = {
        "similarity_hist": fig_similarity_hist(tfidf_df, sem_df, hyb_df),
        "corr_heatmap": fig_corr_heatmap(corr_df)
    }

    if return_results:
        return tfidf_df, sem_df, hyb_df, corr_df, figs


# -------------------------------
# CLI Runner
# -------------------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, default=str(DEFAULT_PDF_DIR), help="Folder with PDFs for evaluation")
    ap.add_argument("--topk", type=int, default=5, help="Top-K reviewers to export per model")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pdf_dir = Path(args.pdf_dir)
    run_full_evaluation(pdf_dir=pdf_dir, top_k_export=args.topk)
