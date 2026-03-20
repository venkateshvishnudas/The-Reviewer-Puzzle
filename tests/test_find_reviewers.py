import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from src.find_reviewers import find_top_reviewers, extract_text_from_pdf, clean_text

# Mock data for testing
mock_vectorizer = MagicMock()
mock_vectorizer.transform.return_value = np.array([[0.1, 0.2, 0.3]])

mock_tfidf_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
mock_df = pd.DataFrame({
    'author_id': ['author1', 'author2', 'author3'],
    'title': ['Title1', 'Title2', 'Title3']
})

@pytest.fixture
def mock_loads(monkeypatch):
    monkeypatch.setattr('joblib.load', lambda _: mock_vectorizer)
    monkeypatch.setattr('pandas.read_pickle', lambda _: mock_df)

@pytest.fixture
def mock_cosine_similarity(monkeypatch):
    monkeypatch.setattr('sklearn.metrics.pairwise.cosine_similarity', lambda x, y: np.array([[0.9, 0.8, 0.7]]))

def test_extract_text_from_pdf_success(monkeypatch):
    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock(extract_text=lambda: "Sample text")]
    monkeypatch.setattr('PyPDF2.PdfReader', lambda _: mock_reader)
    text = extract_text_from_pdf("dummy.pdf")
    assert text == "Sample text"

def test_extract_text_from_pdf_failure(monkeypatch):
    monkeypatch.setattr('PyPDF2.PdfReader', lambda _: (_ for _ in ()).throw(Exception("Error")))
    text = extract_text_from_pdf("dummy.pdf")
    assert text == ""

def test_clean_text():
    assert clean_text("Hello, World! 123") == "hello world 123"
    assert clean_text("Text-with-hyphens") == "text-with-hyphens"
    assert clean_text(None) == ""

def test_find_top_reviewers_text_input(mock_loads, mock_cosine_similarity):
    results = find_top_reviewers("This is a sample abstract text.", top_k=2, method="max")
    assert len(results) == 2
    assert results.iloc[0]['author_id'] == 'author1'
    assert results.iloc[0]['similarity'] > results.iloc[1]['similarity']

def test_find_top_reviewers_pdf_input(mock_loads, mock_cosine_similarity, monkeypatch):
    mock_reader = MagicMock()
    mock_reader.pages = [MagicMock(extract_text=lambda: "Sample text from PDF")]
    monkeypatch.setattr('PyPDF2.PdfReader', lambda _: mock_reader)
    results = find_top_reviewers(Path("dummy.pdf"), top_k=2, method="max")
    assert len(results) == 2
    assert results.iloc[0]['author_id'] == 'author1'

def test_find_top_reviewers_invalid_method(mock_loads):
    with pytest.raises(ValueError, match="⚠️ method must be 'max' or 'mean'"):
        find_top_reviewers("This is a sample abstract text.", top_k=2, method="invalid")

def test_find_top_reviewers_short_text(mock_loads):
    with pytest.raises(ValueError, match="❌ Query text too short. Provide a full abstract or content."):
        find_top_reviewers("Short text", top_k=2, method="max")