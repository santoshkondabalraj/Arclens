from __future__ import annotations

import pathlib
import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from yt_rag.config import settings


def build_bm25_retriever(docs: list[Document], k: int | None = None) -> BM25Retriever:
    """Build a fresh BM25 index from the given documents."""
    k = k or settings.sparse_top_k
    retriever = BM25Retriever.from_documents(docs, k=k)
    return retriever


def _index_path(playlist_id: str) -> pathlib.Path:
    return pathlib.Path(settings.bm25_index_dir) / f"{playlist_id}.pkl"


def persist_bm25(retriever: BM25Retriever, playlist_id: str) -> None:
    """Pickle a BM25Retriever to disk."""
    path = _index_path(playlist_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(retriever, fh)


def load_bm25(playlist_id: str) -> BM25Retriever | None:
    """Load a persisted BM25Retriever from disk; returns None if not found."""
    path = _index_path(playlist_id)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def build_and_persist_bm25(
    docs: list[Document], playlist_id: str, k: int | None = None
) -> BM25Retriever:
    """Build, persist, and return a BM25 index for a playlist."""
    retriever = build_bm25_retriever(docs, k=k)
    persist_bm25(retriever, playlist_id)
    return retriever
