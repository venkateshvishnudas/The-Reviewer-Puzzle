import pytest
from unittest.mock import patch, mock_open
import pandas as pd
from pathlib import Path
import json
from src.load_corpus import load_meta, load_parsed_corpus

# Mock data for meta_log.csv
META_LOG_CONTENT = """author_id,paper_id,pdf_path,json_path,status
1,101,/path/to/pdf1,/path/to/json1,success
2,102,/path/to/pdf2,/path/to/json2,failed
3,103,/path/to/pdf3,/path/to/json3,success
"""

# Mock JSON data
JSON_CONTENT = json.dumps({
    "title": "Sample Title",
    "abstract": "Sample Abstract",
    "body": "Sample Body"
})

@pytest.fixture
def mock_meta_log():
    with patch("builtins.open", mock_open(read_data=META_LOG_CONTENT)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_json_file():
    with patch("builtins.open", mock_open(read_data=JSON_CONTENT)) as mock_file:
        yield mock_file

@pytest.fixture
def mock_path_exists():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

def test_load_meta_success(mock_meta_log):
    df = load_meta(Path("dummy/path/meta_log.csv"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Only two entries with 'success' status
    assert all(col in df.columns for col in ["author_id", "paper_id", "json_path", "status"])

def test_load_meta_malformed():
    malformed_content = "author_id,paper_id\n1,101\n2,102,extra_column"
    with patch("builtins.open", mock_open(read_data=malformed_content)):
        df = load_meta(Path("dummy/path/meta_log.csv"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # Should skip malformed line

def test_load_parsed_corpus_success(mock_meta_log, mock_json_file, mock_path_exists):
    df = load_parsed_corpus(Path("dummy/path/meta_log.csv"))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Two valid JSON entries
    assert all(col in df.columns for col in ["author_id", "paper_id", "title", "abstract", "body", "text", "path"])

def test_load_parsed_corpus_missing_json(mock_meta_log):
    with patch("pathlib.Path.exists", return_value=False):
        df = load_parsed_corpus(Path("dummy/path/meta_log.csv"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0  # No valid JSON entries

def test_load_parsed_corpus_tiny_document(mock_meta_log, mock_path_exists):
    tiny_json_content = json.dumps({
        "title": "Short",
        "abstract": "",
        "body": ""
    })
    with patch("builtins.open", mock_open(read_data=tiny_json_content)):
        df = load_parsed_corpus(Path("dummy/path/meta_log.csv"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two valid JSON entries