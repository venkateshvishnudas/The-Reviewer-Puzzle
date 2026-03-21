# test_fastwmd_utils.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.fastwmd_utils import (
    _internet_connected,
    _try_load,
    _load_fasttext_or_glove,
    _get_w2v,
    _mean_emb,
    _require_corpus,
    build_or_load_fastwmd_corpus,
    encode_query_fastwmd,
    CORPUS_PKL,
    FASTWMD_DIR
)
from pathlib import Path

@pytest.fixture
def mock_corpus_pkl(tmp_path):
    data = {
        "title": ["Title 1", "Title 2"],
        "abstract": ["Abstract 1", "Abstract 2"],
        "body": ["Body 1", "Body 2"],
        "text": ["Text 1", "Text 2"]
    }
    df = pd.DataFrame(data)
    pkl_path = tmp_path / "parsed_corpus.pkl"
    df.to_pickle(pkl_path)
    return pkl_path

@pytest.fixture
def mock_fastwmd_dir(tmp_path):
    return tmp_path / "fastwmd_index"

def test_internet_connected():
    with patch('socket.create_connection', return_value=True):
        assert _internet_connected() is True

    with patch('socket.create_connection', side_effect=OSError):
        assert _internet_connected() is False

def test_try_load_success():
    mock_model = MagicMock()
    with patch('gensim.downloader.api.load', return_value=mock_model):
        model = _try_load("mock-model")
        assert model is mock_model

def test_try_load_failure():
    with patch('gensim.downloader.api.load', side_effect=Exception("Load failed")):
        model = _try_load("mock-model")
        assert model is None

def test_load_fasttext_or_glove_no_internet():
    with patch('src.fastwmd_utils._internet_connected', return_value=False):
        with pytest.raises(RuntimeError):
            _load_fasttext_or_glove()

def test_load_fasttext_or_glove_success():
    mock_model = MagicMock()
    with patch('src.fastwmd_utils._internet_connected', return_value=True):
        with patch('gensim.downloader.api.load', return_value=mock_model):
            model, dim = _load_fasttext_or_glove()
            assert model is mock_model
            assert dim == 300

def test_get_w2v():
    mock_model = MagicMock()
    with patch('src.fastwmd_utils._load_fasttext_or_glove', return_value=(mock_model, 300)):
        model, dim = _get_w2v()
        assert model is mock_model
        assert dim == 300

def test_mean_emb():
    mock_w2v = {'word': np.array([1.0, 2.0, 3.0])}
    tokens = ['word', 'unknown']
    result = _mean_emb(tokens, mock_w2v, 3)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

def test_mean_emb_empty():
    mock_w2v = {}
    tokens = ['unknown']
    result = _mean_emb(tokens, mock_w2v, 3)
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))

def test_require_corpus_missing(mock_corpus_pkl):
    with patch('src.fastwmd_utils.CORPUS_PKL', mock_corpus_pkl):
        df = _require_corpus()
        assert not df.empty

def test_require_corpus_file_not_found():
    with patch('src.fastwmd_utils.CORPUS_PKL', Path("non_existent.pkl")):
        with pytest.raises(FileNotFoundError):
            _require_corpus()

def test_build_or_load_fastwmd_corpus(mock_corpus_pkl, mock_fastwmd_dir):
    with patch('src.fastwmd_utils.CORPUS_PKL', mock_corpus_pkl):
        with patch('src.fastwmd_utils.FASTWMD_DIR', mock_fastwmd_dir):
            with patch('src.fastwmd_utils._get_w2v', return_value=(MagicMock(), 300)):
                df, embs = build_or_load_fastwmd_corpus()
                assert len(df) == 2
                assert embs.shape[0] == 2

def test_encode_query_fastwmd():
    mock_w2v = {'word': np.array([1.0, 2.0, 3.0])}
    with patch('src.fastwmd_utils._get_w2v', return_value=(mock_w2v, 3)):
        result = encode_query_fastwmd("word unknown")
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))