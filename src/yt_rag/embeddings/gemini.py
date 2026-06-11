from __future__ import annotations

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from yt_rag.config import settings

try:
    from google.api_core.exceptions import ResourceExhausted
    _RETRY_EXC = (ResourceExhausted, ConnectionError, TimeoutError)
except ImportError:
    _RETRY_EXC = (ConnectionError, TimeoutError)


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Document-side embeddings (task_type=retrieval_document)."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
        task_type="retrieval_document",
    )


def build_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Query-side embeddings (task_type=retrieval_query)."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
        task_type="retrieval_query",
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRY_EXC),
    reraise=True,
)
def _embed_batch(embeddings: GoogleGenerativeAIEmbeddings, texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)


def embed_documents_batched(
    texts: list[str],
    embeddings: GoogleGenerativeAIEmbeddings | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed texts in batches, retrying on rate-limit errors."""
    if embeddings is None:
        embeddings = build_embeddings()
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results.extend(_embed_batch(embeddings, batch))
    return results
