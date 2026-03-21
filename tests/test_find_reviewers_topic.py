import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from src.find_reviewers_topic import (
    clean_text,
    extract_text_from_pdf,
    safe_load,
    find_top_reviewers_topic,
)

# Test clean_text function
def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("123-456") == "123-456"
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("Special $%^&*() characters!") == "special characters"

# Test extract_text_from_pdf function
@patch("src.find_reviewers_topic.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader):
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Sample text")]
    result = extract_text_from_pdf(Path("dummy.pdf"))
    assert result == "sample text"

    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: None)]
    result = extract_text_from_pdf(Path("dummy.pdf"))
    assert result == ""

# Test safe_load function
@patch("src.find_reviewers_topic.joblib.load")
def test_safe_load(mock_joblib_load):
    mock_joblib_load.return_value = "mock_model"
    path = Path("dummy.pkl")
    with patch("pathlib.Path.exists", return_value=True):
        result = safe_load(path, "Mock Model")
        assert result == "mock_model"

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            safe_load(path, "Mock Model")

# Test find_top_reviewers_topic function
@patch("src.find_reviewers_topic.safe_load")
@patch("src.find_reviewers_topic.pd.read_pickle")
def test_find_top_reviewers_topic(mock_read_pickle, mock_safe_load):
    # Mock data
    mock_df = pd.DataFrame({
        "author_id": [1, 2, 3],
        "text": ["text1", "text2", "text3"],
        "topic_vector": [np.array([0.1, 0.2]), np.array([0.2, 0.3]), np.array([0.3, 0.4])]
    })
    mock_read_pickle.return_value = mock_df
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = np.array([[0.1, 0.2]])
    mock_model = MagicMock()
    mock_model.transform.return_value = np.array([[0.1, 0.2]])
    mock_safe_load.side_effect = [mock_model, mock_vectorizer]

    # Test with valid input
    result = find_top_reviewers_topic("Sample text", top_k=2, model="lda")
    assert isinstance(result, pd.DataFrame)
    assert "author_id" in result.columns
    assert "similarity" in result.columns

    # Test with invalid model
    with pytest.raises(ValueError):
        find_top_reviewers_topic("Sample text", model="invalid")

    # Test with short text
    with pytest.raises(ValueError):
        find_top_reviewers_topic("short", model="lda")

    # Test with missing topics_df.pkl
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            find_top_reviewers_topic("Sample text", model="lda")