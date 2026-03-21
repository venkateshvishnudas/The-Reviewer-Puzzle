import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.app import run_model

@pytest.fixture
def mock_find_top_reviewers():
    return pd.DataFrame({
        'reviewer': ['Reviewer1', 'Reviewer2'],
        'score': [0.9, 0.8]
    })

@pytest.fixture
def mock_find_top_reviewers_topic():
    return pd.DataFrame({
        'reviewer': ['Reviewer3', 'Reviewer4'],
        'similarity': [0.85, 0.75]
    })

@pytest.fixture
def mock_hybrid_recommendation():
    return pd.DataFrame({
        'reviewer': ['Reviewer5', 'Reviewer6'],
        'hybrid_similarity': [0.95, 0.88]
    })

@patch('src.app.find_top_reviewers')
@patch('src.app.load_tfidf')
def test_run_model_tfidf(mock_load_tfidf, mock_find_top_reviewers_func, mock_find_top_reviewers):
    mock_load_tfidf.return_value = None
    mock_find_top_reviewers_func.return_value = mock_find_top_reviewers
    result = run_model("TF-IDF", "Sample query text")
    assert not result.empty
    assert 'score' in result.columns
    assert result.iloc[0]['reviewer'] == 'Reviewer1'

@patch('src.app.find_top_reviewers')
@patch('src.app.load_semantic')
def test_run_model_semantic(mock_load_semantic, mock_find_top_reviewers_func, mock_find_top_reviewers):
    mock_load_semantic.return_value = None
    mock_find_top_reviewers_func.return_value = mock_find_top_reviewers
    result = run_model("Semantic (E5)", "Sample query text")
    assert not result.empty
    assert 'score' in result.columns
    assert result.iloc[0]['reviewer'] == 'Reviewer1'

@patch('src.app.find_top_reviewers_topic')
@patch('src.app.load_topic')
def test_run_model_topic(mock_load_topic, mock_find_top_reviewers_topic_func, mock_find_top_reviewers_topic):
    mock_load_topic.return_value = None
    mock_find_top_reviewers_topic_func.return_value = mock_find_top_reviewers_topic
    result = run_model("Topic (LDA/NMF)", "Sample query text")
    assert not result.empty
    assert 'similarity' in result.columns
    assert result.iloc[0]['reviewer'] == 'Reviewer3'

@patch('src.app.hybrid_recommendation')
@patch('src.app.load_hybrid')
def test_run_model_hybrid(mock_load_hybrid, mock_hybrid_recommendation_func, mock_hybrid_recommendation):
    mock_load_hybrid.return_value = None
    mock_hybrid_recommendation_func.return_value = mock_hybrid_recommendation
    result = run_model("Hybrid", "Sample query text", alpha=0.5)
    assert not result.empty
    assert 'hybrid_similarity' in result.columns
    assert result.iloc[0]['reviewer'] == 'Reviewer5'

def test_run_model_invalid_choice():
    result = run_model("Invalid Model", "Sample query text")
    assert result.empty

@patch('src.app.load_tfidf')
def test_run_model_exception_handling(mock_load_tfidf):
    mock_load_tfidf.side_effect = Exception("Test exception")
    result = run_model("TF-IDF", "Sample query text")
    assert result.empty