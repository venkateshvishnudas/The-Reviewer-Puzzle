import sys
from pathlib import Path
import warnings
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadWarning
import matplotlib.pyplot as plt

# -----------------------------------------------
# Suppress PyPDF2 warnings
# -----------------------------------------------
warnings.filterwarnings("ignore", category=PdfReadWarning)

# -----------------------------------------------
# Ensure src/ imports work
# -----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
TEST_PDF_DIR = DATA_DIR / "test_pdfs"
EVAL_DIR = DATA_DIR / "eval"
TEST_PDF_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------
# Backend Model Wrapper
# -----------------------------------------------
def run_model(model_choice: str, query_text: str, alpha: float = 0.4) -> pd.DataFrame:
    results = pd.DataFrame()
    try:
        if model_choice == "TF-IDF":
            from src.find_reviewers import find_top_reviewers
            results = find_top_reviewers(query_text, top_k=5)
            if "score" in results.columns:
                results.rename(columns={"score": "similarity"}, inplace=True)

        elif model_choice == "Semantic (E5)":
            from src.find_reviewers_semantic import find_top_reviewers
            results = find_top_reviewers(query_text, top_k=5)

        elif model_choice == "Topic (LDA/NMF)":
            from src.find_reviewers_topic import find_top_reviewers_topic
            results = find_top_reviewers_topic(query_text, top_k=5, model="lda")
            if results is None:
                results = pd.DataFrame()

        elif model_choice == "Hybrid":
            from src.find_reviewers_hybrid import hybrid_recommendation
            results = hybrid_recommendation(query_text, alpha=alpha, top_k=5)
            if "hybrid_similarity" in results.columns:
                results.rename(columns={"hybrid_similarity": "similarity"}, inplace=True)
        else:
            st.error("⚠️ Unknown model selected.")
    except Exception as e:
        st.error(f"❌ Error while running {model_choice}: {e}")
    return results

# -----------------------------------------------
# Streamlit UI Setup
# -----------------------------------------------
st.set_page_config(
    page_title="Reviewer Recommendation Engine",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Reviewer Recommendation Engine")
st.caption("Find suitable reviewers using TF-IDF, Semantic, Topic, and Hybrid models.")
st.markdown("---")

# -----------------------------------------------
# 1️⃣ Upload or Paste Query
# -----------------------------------------------
st.subheader("1️⃣ Upload or Paste Paper Text")

uploaded_pdf = st.file_uploader("📎 Upload Research Paper (PDF)", type=["pdf"])
query_text = st.text_area(
    "📝 Or paste the abstract / main text here:",
    placeholder="Enter abstract or main content...",
    height=200,
)

# Extract text if PDF uploaded
if uploaded_pdf is not None:
    save_path = TEST_PDF_DIR / uploaded_pdf.name
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"📁 Saved to: {save_path}")

    reader = PdfReader(uploaded_pdf)
    extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    if not query_text.strip():
        query_text = extracted_text
        st.success("✅ Text extracted from uploaded PDF!")

    with st.expander("📘 Preview Extracted Text"):
        preview = extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else "")
        st.write(preview)

# -----------------------------------------------
# 2️⃣ Choose Model
# -----------------------------------------------
st.subheader("2️⃣ Choose the NLP Model")

model_choice = st.selectbox(
    "Select Model:",
    ["TF-IDF", "Semantic (E5)", "Topic (LDA/NMF)", "Hybrid"],
)

alpha = 0.4
if model_choice == "Hybrid":
    alpha = st.slider("Adjust Hybrid Weight (TF-IDF ↔ Semantic)", 0.0, 1.0, 0.4, 0.1)

# -----------------------------------------------
# 3️⃣ Run Matching
# -----------------------------------------------
st.markdown("---")
col_run, col_eval = st.columns(2)

with col_run:
    run_clicked = st.button("🔍 Find Top Reviewers")

if run_clicked:
    if not query_text or len(query_text.strip()) < 30:
        st.warning("⚠️ Please upload a paper or enter a full abstract.")
    else:
        with st.spinner(f"Running {model_choice} model... ⏳"):
            results = run_model(model_choice, query_text, alpha)
        st.success("✅ Matching complete!")

        if not results.empty:
            st.subheader("🎯 Top Matching Reviewers")

            # Sort and format similarity column
            results_sorted = results.sort_values("similarity", ascending=False).reset_index(drop=True)
            results_sorted["similarity"] = results_sorted["similarity"].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x
            )

            # Enhanced DataFrame styling for visibility
            def highlight_best(s):
                is_max = s == s.max()
                return ['background-color: #1E40AF; color: white; font-weight: bold;' if v else '' for v in is_max]

            styled_df = (
                results_sorted.style
                .apply(highlight_best, subset=["similarity"], axis=0)
                .set_table_styles(
                    [
                        {"selector": "thead th", "props": [("background-color", "#1E293B"), ("color", "white"), ("font-weight", "bold")]},
                        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#F1F5F9")]},
                        {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#E2E8F0")]},
                    ]
                )
            )

            st.dataframe(styled_df, use_container_width=True)

        else:
            st.info("ℹ️ No matching reviewers found for this input.")

# -----------------------------------------------
# 4️⃣ Full Batch Evaluation (no F1/P/R)
# -----------------------------------------------
with col_eval:
    eval_clicked = st.button("🧪 Evaluate All Models")

if eval_clicked:
    from src.evaluate_models import run_full_evaluation

    with st.spinner("Running evaluation pipeline... ⏳"):
        tfidf_df, sem_df, hyb_df, corr_df, figs = run_full_evaluation(
            k=5, top_k_export=10, return_results=True, show_plots=False
        )

    st.success("✅ Evaluation complete!")

    # Show correlation matrix
    st.subheader("📈 Model-to-Model Correlation Matrix")
    st.dataframe(
        corr_df.style.highlight_max(axis=None, color="#60A5FA")
        .set_table_styles(
            [
                {"selector": "thead th", "props": [("background-color", "#1E3A8A"), ("color", "white"), ("font-weight", "bold")]},
                {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#F1F5F9")]},
                {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#E2E8F0")]},
            ]
        ),
        use_container_width=True,
    )

    # Similarity distributions
    st.subheader("📊 Similarity Distributions")
    st.pyplot(figs["similarity_hist"])

    # Correlation heatmap
    st.pyplot(figs["corr_heatmap"])

    # Timestamped CSV downloads
    st.subheader("⬇️ Download Evaluation CSVs (Timestamped)")
    for name, df in [("TF-IDF", tfidf_df), ("Semantic", sem_df), ("Hybrid", hyb_df)]:
        st.download_button(
            label=f"Download {name} Results (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{name.lower()}_results_latest.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("© 2025 Reviewer Recommender — Streamlit + Transformers + Gensim + Matplotlib")
