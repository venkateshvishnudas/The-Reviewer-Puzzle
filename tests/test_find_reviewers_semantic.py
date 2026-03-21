import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from src.find_reviewers_semantic import (
    clean_text,
    extract_text_from_pdf,
    find_top_reviewers,
)

# Mock data for testing
mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
mock_meta_df = pd.DataFrame({
    "author_id": ["author_1", "author_2"],
    "similarity": [0.0, 0.0]
})

@pytest.fixture
def mock_embedding_files(tmp_path):
    emb_path = tmp_path / "doc_embeddings.npy"
    meta_path = tmp_path / "embeddings_df.pkl"
    np.save(emb_path, mock_embeddings)
    mock_meta_df.to_pickle(meta_path)
    return emb_path, meta_path

def test_clean_text():
    assert clean_text("Hello  World") == "Hello World"
    assert clean_text("   Leading and trailing spaces   ") == "Leading and trailing spaces"
    assert clean_text("\u00A0Non-breaking space") == "Non-breaking space"
    assert clean_text(None) == ""

@patch("src.find_reviewers_semantic.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader):
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=MagicMock(return_value="Page 1 text"))]
    result = extract_text_from_pdf("dummy.pdf")
    assert result == "Page 1 text"

@patch("src.find_reviewers_semantic.PdfReader")
def test_extract_text_from_pdf_error(mock_pdf_reader):
    mock_pdf_reader.side_effect = Exception("PDF read error")
    result = extract_text_from_pdf("dummy.pdf")
    assert result == ""

@patch("src.find_reviewers_semantic.SentenceTransformer")
@patch("src.find_reviewers_semantic.np.load")
@patch("src.find_reviewers_semantic.pd.read_pickle")
def test_find_top_reviewers(mock_read_pickle, mock_np_load, mock_sentence_transformer, mock_embedding_files):
    emb_path, meta_path = mock_embedding_files
    mock_np_load.return_value = mock_embeddings
    mock_read_pickle.return_value = mock_meta_df
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_sentence_transformer.return_value = mock_model

    result = find_top_reviewers("Sample text", top_k=1)
    assert len(result) == 1
    assert "author_id" in result.columns
    assert "similarity" in result.columns

@patch("src.find_reviewers_semantic.np.load")
@patch("src.find_reviewers_semantic.pd.read_pickle")
def test_find_top_reviewers_missing_embeddings(mock_read_pickle, mock_np_load):
    mock_np_load.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        find_top_reviewers("Sample text")

def test_find_top_reviewers_short_query():
    with pytest.raises(ValueError):
        find_top_reviewers("Too short")