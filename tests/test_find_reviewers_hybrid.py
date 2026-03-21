import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
from src.find_reviewers_hybrid import clean_text, extract_text_from_pdf, hybrid_recommendation

# Mock data for testing
mock_tfidf_matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
mock_corpus_embs = np.array([[0.5, 0.6], [0.7, 0.8]])
mock_vectorizer = MagicMock()
mock_vectorizer.transform.return_value = np.array([[0.1, 0.2]])
mock_model = MagicMock()
mock_model.encode.return_value = np.array([[0.5, 0.6]])

# Mock dataframes
mock_tfidf_df = pd.DataFrame({'paper_id': [1, 2], 'author_id': ['A1', 'A2']})
mock_emb_df = pd.DataFrame({'paper_id': [1, 2], 'author_id': ['A1', 'A2']})

@pytest.fixture
def setup_mocks():
    with patch('src.find_reviewers_hybrid.joblib.load', side_effect=[mock_vectorizer, mock_tfidf_matrix, mock_tfidf_df]), \
         patch('src.find_reviewers_hybrid.np.load', return_value=mock_corpus_embs), \
         patch('src.find_reviewers_hybrid.pd.read_pickle', side_effect=[mock_tfidf_df, mock_emb_df]), \
         patch('src.find_reviewers_hybrid.SentenceTransformer', return_value=mock_model):
        yield

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("123-456") == "123-456"
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_extract_text_from_pdf(mocker):
    mock_pdf_reader = mocker.patch('src.find_reviewers_hybrid.PdfReader')
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Sample text")]
    assert extract_text_from_pdf("dummy.pdf") == "sample text"

def test_hybrid_recommendation_valid_input(setup_mocks):
    result = hybrid_recommendation("Sample query text", alpha=0.5, top_k=1)
    assert isinstance(result, pd.DataFrame)
    assert 'author_id' in result.columns
    assert 'similarity' in result.columns

def test_hybrid_recommendation_short_text(setup_mocks):
    with pytest.raises(ValueError, match="❌ Query text too short."):
        hybrid_recommendation("Short text", alpha=0.5, top_k=1)

def test_hybrid_recommendation_pdf_input(setup_mocks, mocker):
    mock_extract_text = mocker.patch('src.find_reviewers_hybrid.extract_text_from_pdf', return_value="Sample query text")
    result = hybrid_recommendation(Path("dummy.pdf"), alpha=0.5, top_k=1)
    mock_extract_text.assert_called_once_with(Path("dummy.pdf"))
    assert isinstance(result, pd.DataFrame)

def test_hybrid_recommendation_invalid_alpha(setup_mocks):
    with pytest.raises(ValueError):
        hybrid_recommendation("Sample query text", alpha=1.5, top_k=1)

def test_hybrid_recommendation_invalid_top_k(setup_mocks):
    with pytest.raises(ValueError):
        hybrid_recommendation("Sample query text", alpha=0.5, top_k=-1)