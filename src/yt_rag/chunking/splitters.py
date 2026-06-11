from __future__ import annotations

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from yt_rag.config import settings

_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def build_body_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.body_chunk_size,
        chunk_overlap=settings.body_chunk_overlap,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def build_meta_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.meta_chunk_size,
        chunk_overlap=0,
        length_function=_token_len,
        separators=["\n\n", "\n", " "],
    )


def split_documents(docs: list[Document]) -> list[Document]:
    """Two-tier chunking: route body vs. metadata docs to separate splitters.

    Output chunks carry all parent metadata fields plus chunk_index and
    token_count.  chunk_type is normalized to "body" or "metadata".
    """
    body_splitter = build_body_splitter()
    meta_splitter = build_meta_splitter()

    body_docs = [d for d in docs if d.metadata.get("chunk_type") != "metadata"]
    meta_docs = [d for d in docs if d.metadata.get("chunk_type") == "metadata"]

    output: list[Document] = []

    # Track per-video chunk index to number chunks within each source
    video_counters: dict[str, int] = {}

    def _enrich(chunks: list[Document], chunk_type: str) -> list[Document]:
        enriched = []
        for chunk in chunks:
            vid = chunk.metadata.get("video_id", "unknown")
            idx = video_counters.get(vid, 0)
            video_counters[vid] = idx + 1
            chunk.metadata["chunk_type"] = chunk_type
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["token_count"] = _token_len(chunk.page_content)
            enriched.append(chunk)
        return enriched

    if body_docs:
        body_chunks = body_splitter.split_documents(body_docs)
        output.extend(_enrich(body_chunks, "body"))

    if meta_docs:
        meta_chunks = meta_splitter.split_documents(meta_docs)
        output.extend(_enrich(meta_chunks, "metadata"))

    return output
