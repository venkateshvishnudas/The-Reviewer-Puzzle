Reviewer Recommendation System
🚀 An AI-powered engine to match research papers with the most suitable reviewers

Built with: TF-IDF • Sentence-Transformers (E5-base) • Streamlit • Scikit-learn • Matplotlib

Please find the attached deployed link: https://huggingface.co/spaces/venkateshvish/The-Reviewer-Puzzle

📘 Overview

Finding the right reviewers for research submissions can be complex, especially in interdisciplinary fields.
This system automates the process by analyzing research papers and recommending reviewers whose prior publications align most closely with the paper’s content.

The app supports three intelligent recommendation models:

TF-IDF Model — Lexical similarity using traditional text frequency analysis

Semantic Model (E5) — Deep contextual understanding using transformer embeddings

Hybrid Model — Combines TF-IDF and E5 similarity for balanced precision and recall

Key Features

Upload a research paper (PDF or text)
Get top-K reviewer matches instantly
Choose between TF-IDF, Semantic, or Hybrid matching
Visualize similarity distributions and model correlations
Batch evaluation across multiple PDFs
Fully deployable Streamlit app

Installation & Setup
🧰 Prerequisites

Python 3.10+

pip (latest)

👨‍💻 Author

Venkatesh Vishnudas
