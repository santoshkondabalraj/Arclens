"""CLI: ingest a YouTube playlist into the RAG index.

Usage:
    python scripts/ingest_playlist.py <PLAYLIST_URL> [--user USER_ID] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse

from dotenv import load_dotenv

load_dotenv()

from yt_rag.chunking.splitters import split_documents
from yt_rag.ingestion.cleaner import clean_document
from yt_rag.ingestion.freshness import record_ingestion, staleness_warning
from yt_rag.ingestion.loader import YouTubePlaylistLoader
from yt_rag.retrieval.bm25 import build_and_persist_bm25
from yt_rag.retrieval.store import ensure_index_exists, get_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a YouTube playlist")
    parser.add_argument("playlist_url", help="Full YouTube playlist URL")
    parser.add_argument("--user", default="default_user", help="User ID for namespace")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cleaned docs and chunk counts without writing to Pinecone",
    )
    args = parser.parse_args()

    qs = urllib.parse.parse_qs(urllib.parse.urlparse(args.playlist_url).query)
    playlist_id = qs.get("list", ["unknown"])[0]

    warning = staleness_warning(playlist_id)
    if warning:
        print(f"[WARN] {warning}", file=sys.stderr)

    print(f"Loading transcripts from: {args.playlist_url}")
    loader = YouTubePlaylistLoader(args.playlist_url)
    raw_docs = list(loader.lazy_load())
    meta_docs = loader.load_metadata_chunks()
    print(f"Loaded {len(raw_docs)} transcript docs + {len(meta_docs)} metadata docs")

    cleaned = [clean_document(d) for d in raw_docs + meta_docs]
    chunks = split_documents(cleaned)

    body_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "body"]
    meta_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "metadata"]
    print(f"Chunks: {len(body_chunks)} body, {len(meta_chunks)} metadata")

    if args.dry_run:
        print("\n--- DRY RUN: first 3 body chunks ---")
        for i, chunk in enumerate(body_chunks[:3]):
            print(f"\n[{i+1}] {chunk.metadata}")
            print(chunk.page_content[:300])
        return

    ensure_index_exists()
    namespace = f"{args.user}_{playlist_id}"
    vectorstore = get_vectorstore(namespace)
    vectorstore.add_documents(chunks)
    print(f"Upserted {len(chunks)} chunks to Pinecone namespace '{namespace}'")

    build_and_persist_bm25(chunks, playlist_id)
    print(f"BM25 index persisted for playlist '{playlist_id}'")

    record_ingestion(playlist_id, args.user)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
