from __future__ import annotations

import re

from langchain_core.documents import Document
from langsmith import traceable

from yt_rag.config import settings
from yt_rag.generation.chain import build_generation_chain
from yt_rag.graph.state import RAGState
from yt_rag.retrieval.bm25 import load_bm25
from yt_rag.retrieval.hybrid import (
    build_dense_retriever,
    build_ensemble_retriever,
)
from yt_rag.retrieval.store import get_vectorstore

_CITATION_RE = re.compile(r'\[Source:\s*"([^"]+)",\s*([^,\]]+),\s*~(\d+)s\]')

# Meta-instructions users add to questions that confuse the cross-encoder because
# the phrases ("according to the video") don't appear in transcript chunks.
_META_PHRASE_RE = re.compile(
    r"\b(?:according to|as (?:mentioned|discussed|stated|explained) in|"
    r"(?:in|from|per|based on)) (?:the |this )?(?:video|lecture|podcast|episode|clip|transcript|content)s?\b",
    re.IGNORECASE,
)

@traceable(name="retrieve")
def retrieve(state: RAGState) -> dict:
    namespace = f"{state['user_id']}_{state['playlist_id']}"
    vectorstore = get_vectorstore(namespace)
    bm25 = load_bm25(state["playlist_id"])

    dense = build_dense_retriever(vectorstore, state.get("metadata_filter"))

    # Strip meta-phrases ("according to the video", "from the video") — they have no
    # transcript vocabulary and add noise to both embedding similarity and BM25 scoring.
    query = _META_PHRASE_RE.sub("", state["question"]).strip() or state["question"]
    dense_docs = dense.invoke(query)

    if bm25 is not None:
        bm25_docs = bm25.invoke(query)
        ensemble = build_ensemble_retriever(dense, bm25)
        docs = ensemble.weighted_reciprocal_rank([dense_docs, bm25_docs])
    else:
        docs = dense_docs

    return {"retrieved_docs": docs}


@traceable(name="rerank")
def rerank(state: RAGState) -> dict:
    docs = state["retrieved_docs"]
    if not docs:
        return {"reranked_docs": []}

    from yt_rag.retrieval.hybrid import _get_cross_encoder
    encoder = _get_cross_encoder()
    # Use original question (meta-phrases stripped) — ms-marco was trained on NL query-passage
    # pairs, not keyword bags. The rewrite is BM25-only; the cross-encoder scores better
    # against full natural-language questions.
    normalized_q = _META_PHRASE_RE.sub("", state["question"]).strip() or state["question"]
    pairs = [(normalized_q, doc.page_content) for doc in docs]
    scores = encoder.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top = scored[: settings.rerank_top_n]

    reranked = []
    for score, doc in top:
        copy = Document(page_content=doc.page_content, metadata={**doc.metadata})
        copy.metadata["relevance_score"] = float(score)
        reranked.append(copy)

    return {"reranked_docs": reranked}


@traceable(name="threshold_check")
def threshold_check(state: RAGState) -> dict:
    min_score = state.get("min_rerank_score", settings.min_rerank_score)
    docs = state["reranked_docs"]
    passing = [
        d for d in docs
        if d.metadata.get("relevance_score", 0.0) >= min_score
    ]
    if not passing:
        return {
            "threshold_passed": False,
            "refusal_reason": "No chunks exceeded the confidence threshold. Cannot provide a grounded answer.",
        }
    return {"threshold_passed": True, "reranked_docs": passing}


@traceable(name="generate")
def generate(state: RAGState) -> dict:
    chain = build_generation_chain()
    reranked = state["reranked_docs"]
    answer = chain.invoke({
        "docs": reranked,
        "question": state["question"],
    })
    # Explicitly carry reranked_docs forward so format_citations always sees them.
    # Without this, LangGraph may revert the field to its initial [] between nodes.
    return {"answer": answer, "reranked_docs": reranked}


def _find_doc_for_title(title: str, title_to_doc: dict):
    """Exact match first, then case-insensitive substring, then first available."""
    if title in title_to_doc:
        return title_to_doc[title]
    tl = title.lower()
    for k, d in title_to_doc.items():
        if tl in k.lower() or k.lower() in tl:
            return d
    # Last resort: return first doc that has a video_id
    for d in title_to_doc.values():
        if d.metadata.get("video_id"):
            return d
    return None


@traceable(name="format_citations")
def format_citations(state: RAGState) -> dict:
    import json as _json

    answer = state.get("answer", "") or ""

    # Detect when the LLM itself outputs the JSON refusal format rather than an answer.
    stripped = answer.strip()
    if stripped.startswith("{"):
        try:
            parsed = _json.loads(stripped)
            if isinstance(parsed, dict) and parsed.get("refusal"):
                final_response = {
                    "refusal": True,
                    "reason": parsed.get("reason", "Insufficient context in the corpus."),
                }
                return {"citations": [], "final_response": final_response}
        except (_json.JSONDecodeError, ValueError):
            pass

    # Use reranked_docs; fall back to retrieved_docs if the list is somehow empty.
    candidate_docs = (
        state.get("reranked_docs")
        or state.get("retrieved_docs")
        or []
    )
    title_to_doc = {
        doc.metadata.get("title", ""): doc
        for doc in candidate_docs
    }
    citations = []
    for match in _CITATION_RE.finditer(answer):
        title = match.group(1)
        doc = _find_doc_for_title(title, title_to_doc)
        citations.append({
            "title": title,
            "channel": match.group(2).strip(),
            "timestamp_seconds": int(match.group(3)),
            "video_id": doc.metadata.get("video_id", "") if doc else "",
        })
    # Deduplicate by (video_id, timestamp_seconds) — keep first occurrence
    seen: set[tuple] = set()
    unique: list[dict] = []
    for c in citations:
        key = (c["video_id"], c["timestamp_seconds"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    citations = unique

    final_response = {
        "answer": answer,
        "citations": citations,
    }
    return {"citations": citations, "final_response": final_response}


@traceable(name="refusal")
def refusal(state: RAGState) -> dict:
    final_response = {
        "refusal": True,
        "reason": state.get("refusal_reason") or "Insufficient context in the corpus.",
    }
    return {"final_response": final_response}
