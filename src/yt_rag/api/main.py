from __future__ import annotations

import asyncio
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from yt_rag.api.dependencies import get_rag_graph
from yt_rag.api.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    IngestRequest,
    StatusResponse,
)
from yt_rag.config import settings
from yt_rag.ingestion.freshness import (
    list_all_ingested,
    load_ingestion_meta,
    record_ingestion,
    staleness_warning,
)

app = FastAPI(title="YouTube Playlist RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/ui/index.html")


@app.on_event("startup")
async def warmup() -> None:
    """Pre-load the cross-encoder model so the first /chat request isn't cold."""
    from yt_rag.retrieval.hybrid import _get_cross_encoder
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_cross_encoder)


def _run_ingestion_pipeline(playlist_url: str, user_id: str) -> None:
    """Full ingestion pipeline run synchronously in a background thread."""
    from yt_rag.chunking.splitters import split_documents
    from yt_rag.ingestion.cleaner import stamp_and_clean_document
    from yt_rag.ingestion.loader import YouTubePlaylistLoader, _extract_playlist_video_ids
    from yt_rag.retrieval.bm25 import build_and_persist_bm25
    from yt_rag.retrieval.store import get_vectorstore, ensure_index_exists

    import urllib.parse
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(playlist_url).query)
    playlist_id = qs.get("list", ["unknown"])[0]
    namespace = f"{user_id}_{playlist_id}"

    ensure_index_exists()

    # Fetch playlist title for the library display
    playlist_title = playlist_id
    try:
        from pytubefix import Playlist
        pl = Playlist(playlist_url)
        playlist_title = pl.title or playlist_id
    except Exception:
        pass

    loader = YouTubePlaylistLoader(playlist_url)
    raw_docs = list(loader.lazy_load())
    meta_docs = loader.load_metadata_chunks()
    all_raw = raw_docs + meta_docs

    # Chunk BEFORE cleaning so [T:Xs] sentinels are still present.
    # stamp_and_clean_document extracts the first sentinel as timestamp_seconds
    # then strips it (and other noise) from the final chunk text.
    raw_chunks = split_documents(all_raw)
    chunks = [stamp_and_clean_document(c) for c in raw_chunks]

    vectorstore = get_vectorstore(namespace)
    vectorstore.add_documents(chunks)

    build_and_persist_bm25(chunks, playlist_id)
    record_ingestion(playlist_id, user_id, playlist_title)


@app.post("/ingest", response_model=StatusResponse)
async def ingest_playlist(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> StatusResponse:
    """Trigger ingestion of a YouTube playlist (runs in background)."""
    import urllib.parse
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(request.playlist_url).query)
    playlist_id = qs.get("list", ["unknown"])[0]

    warning = staleness_warning(playlist_id)

    background_tasks.add_task(
        _run_ingestion_pipeline, request.playlist_url, request.user_id
    )
    return StatusResponse(
        status="ingestion_started",
        message="Ingestion running in background.",
        warning=warning,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    graph=Depends(get_rag_graph),
) -> ChatResponse:
    """Ask a question over an ingested playlist; enforces 8-second hard ceiling."""
    state: dict[str, Any] = {
        "question": request.question,
        "playlist_id": request.playlist_id,
        "user_id": request.user_id,
        "metadata_filter": request.filter,
        "retrieved_docs": [],
        "reranked_docs": [],
        "min_rerank_score": settings.min_rerank_score,
        "threshold_passed": False,
        "refusal_reason": None,
        "answer": None,
        "citations": [],
        "final_response": None,
    }

    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, graph.invoke, state),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        return ChatResponse(
            refusal=True,
            reason="Request exceeded the 30-second latency ceiling.",
        )

    final = result.get("final_response") or {}

    if final.get("refusal"):
        return ChatResponse(refusal=True, reason=final.get("reason"))

    citations = [
        Citation(
            title=c.get("title", ""),
            channel=c.get("channel", ""),
            timestamp_seconds=c.get("timestamp_seconds", 0),
            video_id=c.get("video_id", ""),
        )
        for c in final.get("citations", [])
    ]
    return ChatResponse(
        answer=final.get("answer"),
        citations=citations,
    )


@app.post("/debug/retrieve")
async def debug_retrieve(request: ChatRequest) -> dict:
    """Return raw retrieval scores — helps calibrate MIN_RERANK_SCORE."""
    from yt_rag.graph.nodes import _META_PHRASE_RE
    from yt_rag.retrieval.bm25 import load_bm25
    from yt_rag.retrieval.hybrid import build_dense_retriever, build_ensemble_retriever, _get_cross_encoder
    from yt_rag.retrieval.store import get_vectorstore

    namespace = f"{request.user_id}_{request.playlist_id}"
    vectorstore = get_vectorstore(namespace)
    bm25 = load_bm25(request.playlist_id)

    dense = build_dense_retriever(vectorstore, request.filter)
    query = _META_PHRASE_RE.sub("", request.question).strip() or request.question

    dense_docs = dense.invoke(query)
    if bm25 is not None:
        bm25_docs = bm25.invoke(query)
        ensemble = build_ensemble_retriever(dense, bm25)
        retrieved = ensemble.weighted_reciprocal_rank([dense_docs, bm25_docs])
    else:
        retrieved = dense_docs

    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in retrieved]
    scores = encoder.predict(pairs)
    scored = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)

    return {
        "question": request.question,
        "retrieved_count": len(retrieved),
        "current_threshold": settings.min_rerank_score,
        "reranked": [
            {
                "score": round(float(score), 4),
                "title": doc.metadata.get("title"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "preview": doc.page_content[:120],
            }
            for score, doc in scored[:10]
        ],
    }


@app.get("/playlists")
async def list_playlists() -> list[dict]:
    """Return all ingested playlists sorted by most recently ingested."""
    return list_all_ingested()


@app.get("/status/{playlist_id}", response_model=StatusResponse)
async def get_status(playlist_id: str, user_id: str = "") -> StatusResponse:
    """Return ingestion status and staleness warning for a playlist."""
    meta = load_ingestion_meta(playlist_id)
    warning = staleness_warning(playlist_id)
    return StatusResponse(
        status=meta.get("status", "not_ingested"),
        last_ingested=meta.get("last_ingested"),
        stale=warning is not None,
        warning=warning,
    )


# Static files must be mounted last so API routes take precedence.
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
