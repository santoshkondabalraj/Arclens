"""Tests for the LangGraph RAG flow.

These tests mock all external services (Pinecone, Gemini, BM25) to verify
graph routing logic in isolation.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from yt_rag.graph.state import RAGState


def _base_state(**overrides) -> dict:
    state: dict = {
        "question": "What is machine learning?",
        "playlist_id": "test_pl",
        "user_id": "test_user",
        "metadata_filter": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "min_rerank_score": 0.3,
        "threshold_passed": False,
        "refusal_reason": None,
        "answer": None,
        "citations": [],
        "final_response": None,
    }
    state.update(overrides)
    return state


def test_threshold_check_passes_with_two_high_score_docs():
    from yt_rag.graph.nodes import threshold_check

    docs = [
        Document(page_content="doc1", metadata={"relevance_score": 0.8}),
        Document(page_content="doc2", metadata={"relevance_score": 0.7}),
    ]
    state = _base_state(reranked_docs=docs)
    result = threshold_check(state)
    assert result["threshold_passed"] is True


def test_threshold_check_fails_with_one_low_score_doc():
    from yt_rag.graph.nodes import threshold_check

    docs = [
        Document(page_content="doc1", metadata={"relevance_score": 0.1}),
    ]
    state = _base_state(reranked_docs=docs)
    result = threshold_check(state)
    assert result["threshold_passed"] is False
    assert "refusal_reason" in result


def test_format_citations_extracts_correctly():
    from yt_rag.graph.nodes import format_citations

    answer = (
        'The guest discussed risks. [Source: "AI Safety Podcast", Lex Fridman, ~1200s] '
        'They also mentioned alignment. [Source: "Future of AI", 80k Hours, ~300s]'
    )
    state = _base_state(answer=answer)
    result = format_citations(state)
    assert len(result["citations"]) == 2
    assert result["citations"][0]["title"] == "AI Safety Podcast"
    assert result["citations"][1]["timestamp_seconds"] == 300


def test_refusal_node_returns_structured_response():
    from yt_rag.graph.nodes import refusal

    state = _base_state(refusal_reason="Not enough context.")
    result = refusal(state)
    assert result["final_response"]["refusal"] is True
    assert "Not enough context" in result["final_response"]["reason"]


def test_graph_routes_to_refusal_when_no_docs():
    from yt_rag.graph.rag_graph import build_rag_graph

    with (
        patch("yt_rag.graph.nodes.get_vectorstore") as mock_vs,
        patch("yt_rag.graph.nodes.load_bm25", return_value=None),
        patch("yt_rag.graph.nodes.build_dense_retriever") as mock_dense,
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_dense.return_value = mock_retriever
        mock_vs.return_value = MagicMock()

        graph = build_rag_graph()
        result = graph.invoke(_base_state())

    final = result["final_response"]
    assert final is not None
    assert final.get("refusal") is True
