from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder

from yt_rag.config import settings

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder_cache: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        _cross_encoder_cache = CrossEncoder(_CROSS_ENCODER_MODEL)
    return _cross_encoder_cache


def build_dense_retriever(
    vectorstore,
    metadata_filter: dict | None = None,
) -> BaseRetriever:
    search_kwargs: dict = {"k": settings.dense_top_k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def build_ensemble_retriever(
    dense_retriever: BaseRetriever,
    bm25_retriever: BM25Retriever,
    weights: tuple[float, float] = (0.5, 0.5),
) -> EnsembleRetriever:
    """RRF fusion of dense and sparse retrievers.

    EnsembleRetriever uses Reciprocal Rank Fusion internally when c is set.
    c=60 is the standard RRF constant; higher values reduce aggressive re-ranking.
    """
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=list(weights),
        c=60,
    )


def build_hybrid_retriever(
    vectorstore,
    bm25_retriever: BM25Retriever,
    metadata_filter: dict | None = None,
) -> EnsembleRetriever:
    """Dense + sparse RRF fusion retriever.

    Reranking is done separately in the graph's rerank node using
    sentence_transformers.CrossEncoder.predict() for full score control.
    """
    dense = build_dense_retriever(vectorstore, metadata_filter)
    return build_ensemble_retriever(dense, bm25_retriever)
