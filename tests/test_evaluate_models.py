import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
from src.evaluate_models import (
    extract_text_from_pdf,
    _normalize_similarity_columns,
    fig_similarity_hist,
    fig_corr_heatmap,
    export_results
)

# Test extract_text_from_pdf
def test_extract_text_from_pdf_valid_pdf(mocker):
    mock_pdf_reader = mocker.patch('src.evaluate_models.PdfReader')
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "Sample text")]
    text = extract_text_from_pdf(Path("dummy.pdf"))
    assert text == "Sample text"

def test_extract_text_from_pdf_empty_pdf(mocker):
    mock_pdf_reader = mocker.patch('src.evaluate_models.PdfReader')
    mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "")]
    text = extract_text_from_pdf(Path("dummy.pdf"))
    assert text == ""

def test_extract_text_from_pdf_error_handling(mocker):
    mock_pdf_reader = mocker.patch('src.evaluate_models.PdfReader', side_effect=Exception("Read error"))
    text = extract_text_from_pdf(Path("dummy.pdf"))
    assert text == ""

# Test _normalize_similarity_columns
def test_normalize_similarity_columns():
    df = pd.DataFrame({'score': [0.1, 0.2, 0.3]})
    normalized_df = _normalize_similarity_columns(df)
    assert 'similarity' in normalized_df.columns
    assert normalized_df['similarity'].equals(df['score'])

def test_normalize_similarity_columns_no_change():
    df = pd.DataFrame({'similarity': [0.1, 0.2, 0.3]})
    normalized_df = _normalize_similarity_columns(df)
    assert 'similarity' in normalized_df.columns
    assert normalized_df.equals(df)

# Test fig_similarity_hist
def test_fig_similarity_hist():
    df = pd.DataFrame({'similarity': [0.1, 0.2, 0.3]})
    fig = fig_similarity_hist(df, df, df, df)
    assert fig is not None

# Test fig_corr_heatmap
def test_fig_corr_heatmap():
    corr_df = pd.DataFrame({
        'Model1': [1.0, 0.8],
        'Model2': [0.8, 1.0]
    }, index=['Model1', 'Model2'])
    fig = fig_corr_heatmap(corr_df)
    assert fig is not None

# Test export_results
def test_export_results_no_pdfs(mocker):
    mocker.patch('src.evaluate_models.Path.glob', return_value=[])
    with pytest.raises(FileNotFoundError):
        export_results(Path("dummy_dir"))

def test_export_results_with_pdfs(mocker):
    mock_pdf = MagicMock()
    mock_pdf.stem = "test_pdf"
    mocker.patch('src.evaluate_models.Path.glob', return_value=[mock_pdf])
    mocker.patch('src.evaluate_models.extract_text_from_pdf', return_value="Sample text")
    mocker.patch('src.evaluate_models.tfidf_model', return_value=pd.DataFrame({'similarity': [0.1]}))
    mocker.patch('src.evaluate_models.semantic_model', return_value=pd.DataFrame({'similarity': [0.2]}))
    mocker.patch('src.evaluate_models.hybrid_recommendation', return_value=pd.DataFrame({'similarity': [0.3]}))
    mocker.patch('src.evaluate_models.find_top_reviewers_topic', return_value=pd.DataFrame({'similarity': [0.4]}))
    export_results(Path("dummy_dir"))