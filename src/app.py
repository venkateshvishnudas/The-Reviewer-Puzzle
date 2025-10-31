"""
app.py
======
Final Production-Ready Version (2025-10)
----------------------------------------
✅ Streamlit + Hugging Face–compatible
✅ Prevents model reload on every click
✅ Keeps uploaded PDFs & text in session
✅ Uses temporary storage for runtime safety
✅ Evaluates TF-IDF, Semantic, Topic (LDA/NMF), and Hybrid
"""

import sys
import tempfile
import warnings
from pathlib import Path
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadWarning

# -----------------------------------------------
# 🧹 Suppress Warnings
# -----------------------------------------------
warnings.filterwarnings("ignore", category=PdfReadWarning)

# -----------------------------------------------
# 📂 Path Setup
# -----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_PDF_DIR = Path(tempfile.gettempdir()) / "test_pdfs"
EVAL_DIR = DATA_DIR / "eval"

for d in [DATA_DIR, TEST_PDF_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# -----------------------------------------------
# ⚙️ Caching Model Imports
# -----------------------------------------------
@st.cache_resource
def load_tfidf():
    from src.find_reviewers import find_top_reviewers
    return find_top_reviewers

@st.cache_resource
def load_semantic():
    from src.find_reviewers_semantic import find_top_reviewers
    return find_top_reviewers

@st.cache_resource
def load_hybrid():
    from src.find_reviewers_hybrid import hybrid_recommendation
    return hybrid_recommendation

@st.cache_resource
def load_topic():
    from src.find_reviewers_topic import find_top_reviewers_topic
    return find_top_reviewers_topic

# -----------------------------------------------
# 🧠 Run Model Wrapper
# -----------------------------------------------
def run_model(model_choice: str, query_text: str, alpha: float = 0.4) -> pd.DataFrame:
    try:
        if model_choice == "TF-IDF":
            model = load_tfidf()
            results = model(query_text, top_k=5)
            results.rename(columns={"score": "similarity"}, inplace=True)

        elif model_choice == "Semantic (E5)":
            model = load_semantic()
            results = model(query_text, top_k=5)

        elif model_choice == "Topic (LDA/NMF)":
            model = load_topic()
            results = model(query_text, top_k=5, model="lda")
            if results.empty:
                st.warning("⚠️ No topic model results found.")

        elif model_choice == "Hybrid":
            model = load_hybrid()
            results = model(query_text, alpha=alpha, top_k=5)
            results.rename(columns={"hybrid_similarity": "similarity"}, inplace=True)

        else:
            st.error("⚠️ Unknown model selected.")
            return pd.DataFrame()

        return results

    except Exception as e:
        st.error(f"❌ Error while running {model_choice}: {e}")
        return pd.DataFrame()

# -----------------------------------------------
# 💾 Session State (Preserve Input Across Reruns)
# -----------------------------------------------
if "uploaded_pdf" not in st.session_state:
    st.session_state.uploaded_pdf = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# -----------------------------------------------
# 🎨 Streamlit UI
# -----------------------------------------------
st.set_page_config(page_title="Reviewer Recommendation Engine", page_icon="🧠", layout="centered")

st.title("🧠 Reviewer Recommendation Engine")
st.caption("Find reviewers using TF-IDF, Semantic (E5), Topic (LDA/NMF), and Hybrid models.")
st.markdown("---")

# -----------------------------------------------
# 1️⃣ Upload or Paste Text
# -----------------------------------------------
st.subheader("📎 Upload or Paste Paper Text")

uploaded_pdf = st.file_uploader("Upload Research Paper (PDF)", type=["pdf"])
if uploaded_pdf:
    st.session_state.uploaded_pdf = uploaded_pdf

query_text = st.text_area(
    "Or paste the abstract / main text here:",
    value=st.session_state.query_text,
    placeholder="Enter abstract or main content...",
    height=200,
)
st.session_state.query_text = query_text

# Handle PDF upload and extraction
if st.session_state.uploaded_pdf is not None:
    pdf_file = st.session_state.uploaded_pdf
    save_path = TEST_PDF_DIR / pdf_file.name

    try:
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.success(f"✅ Saved PDF → `{save_path.name}`")

        reader = PdfReader(save_path)
        extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])

        if not st.session_state.query_text.strip():
            st.session_state.query_text = extracted_text
            st.info("✅ Text extracted automatically from uploaded PDF!")

        with st.expander("📘 Preview Extracted Text"):
            preview = extracted_text[:1500] + ("..." if len(extracted_text) > 1500 else "")
            st.write(preview)

    except Exception as e:
        st.error(f"❌ PDF read error: {e}")

# Runtime check
st.write("📂 Uploaded PDFs:", [p.name for p in TEST_PDF_DIR.glob("*.pdf")])

# -----------------------------------------------
# 2️⃣ Model Selection
# -----------------------------------------------
st.subheader("⚙️ Choose NLP Model")
with st.form("model_form"):
    model_choice = st.selectbox(
        "Select Model:",
        ["TF-IDF", "Semantic (E5)", "Topic (LDA/NMF)", "Hybrid"],
    )
    alpha = st.slider("Adjust Hybrid Weight (TF-IDF ↔ Semantic)", 0.0, 1.0, 0.4, 0.1) if model_choice == "Hybrid" else 0.4
    run_clicked = st.form_submit_button("🔍 Find Top Reviewers")

# -----------------------------------------------
# 3️⃣ Run Model
# -----------------------------------------------
if run_clicked:
    text_input = st.session_state.query_text
    if not text_input or len(text_input.strip()) < 30:
        st.warning("⚠️ Please upload a paper or enter a valid abstract.")
    else:
        with st.spinner(f"Running {model_choice} model... ⏳"):
            results = run_model(model_choice, text_input, alpha)
        if not results.empty:
            st.success("✅ Matching complete!")
            st.subheader("🎯 Top Matching Reviewers")
            results["similarity"] = results["similarity"].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
            st.dataframe(results, use_container_width=True)
        else:
            st.info("ℹ️ No matching reviewers found.")

# -----------------------------------------------
# 4️⃣ Full Evaluation
# -----------------------------------------------
st.markdown("---")
if st.button("🧪 Evaluate All Models"):
    from src.evaluate_models import run_full_evaluation
    st.info(f"📍 Evaluating PDFs from: `{TEST_PDF_DIR.resolve()}`")
    st.write("📄 Files Detected:", [p.name for p in TEST_PDF_DIR.glob("*.pdf")])

    try:
        with st.spinner("Running evaluation across all models..."):
            tfidf_df, sem_df, hyb_df, topic_df, corr_df, figs = run_full_evaluation(
                pdf_dir=str(TEST_PDF_DIR),
                k=5,
                top_k_export=10,
                return_results=True,
                show_plots=False,
            )

        st.success("✅ Evaluation complete!")

        st.subheader("📈 Model-to-Model Correlation Matrix")
        st.dataframe(corr_df, use_container_width=True)

        st.subheader("📊 Similarity Distributions")
        st.pyplot(figs["similarity_hist"])
        st.pyplot(figs["corr_heatmap"])

        st.subheader("⬇️ Download Evaluation Results")
        for name, df in [
            ("TF-IDF", tfidf_df),
            ("Semantic", sem_df),
            ("Hybrid", hyb_df),
            ("Topic", topic_df),
        ]:
            if not df.empty:
                st.download_button(
                    label=f"Download {name} Results (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name.lower()}_results.csv",
                    mime="text/csv",
                )

    except FileNotFoundError:
        st.error("❌ No PDFs found! Please upload at least one paper before evaluation.")
    except Exception as e:
        st.error(f"❌ Evaluation failed: {e}")

st.markdown("---")
st.caption("© 2025 Reviewer Recommender — Streamlit + Transformers + Gensim + Matplotlib")
