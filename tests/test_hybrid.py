"""Tests for hybrid retrieval assembly."""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def test_build_ensemble_retriever_returns_rrf_retriever():
    from yt_rag.retrieval.hybrid import build_ensemble_retriever
    from langchain_classic.retrievers import EnsembleRetriever

    dense = MagicMock()
    sparse = MagicMock()
    ensemble = build_ensemble_retriever(dense, sparse)
    assert isinstance(ensemble, EnsembleRetriever)


def test_build_hybrid_retriever_returns_compression_retriever():
    from yt_rag.retrieval.hybrid import build_hybrid_retriever
    from langchain_classic.retrievers import ContextualCompressionRetriever

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    mock_bm25 = MagicMock()

    with patch("yt_rag.retrieval.hybrid._get_cross_encoder") as mock_ce:
        mock_ce.return_value = MagicMock()
        retriever = build_hybrid_retriever(mock_vs, mock_bm25)

    assert isinstance(retriever, ContextualCompressionRetriever)
