from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class RAGState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    question: str
    playlist_id: str
    user_id: str
    metadata_filter: Optional[dict]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs: list[Document]   # dense+sparse fusion output (up to 40)
    reranked_docs: list[Document]    # cross-encoder output (up to 5)
    min_rerank_score: float          # threshold for refusal check

    # ── Decision ─────────────────────────────────────────────────────────────
    threshold_passed: bool
    refusal_reason: Optional[str]

    # ── Output ───────────────────────────────────────────────────────────────
    answer: Optional[str]
    citations: list[dict]            # [{title, channel, timestamp_seconds}, ...]
    final_response: Optional[dict]   # {"answer": ..., "citations": [...]} or refusal
