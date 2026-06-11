"""Tier 1 golden dataset generator.

Uses Claude Opus to produce a diverse set of 16 retrieval test examples per
playlist, including deliberate vocabulary-gap (semantic synonym) probes. Called
automatically during ingestion (best-effort) and via scripts/generate_golden_qa.py.

Requires ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
from collections import defaultdict

from yt_rag.retrieval.bm25 import load_bm25

_TESTS_DIR = pathlib.Path("tests")
_MODEL = "claude-opus-4-8"
_MAX_CHUNKS = 20
_CHUNKS_PER_VIDEO = 3

_PROMPT_TEMPLATE = """\
You are building a test suite for a RAG system that answers questions about YouTube playlists.

The RAG pipeline:
- Uses dense embeddings (Gemini text-embedding-004) + BM25 sparse retrieval fused via RRF
- Reranks with a cross-encoder (ms-marco-MiniLM-L-6-v2) using raw logits
- Refuses to answer if no chunk clears the relevance threshold

Below are transcript chunks from the playlist. Generate exactly 16 test examples that \
probe different retrieval behaviours.

TRANSCRIPT CHUNKS:
{chunks}

Generate exactly 16 examples as a JSON array. Each object must have:
  - "question": the probe question (natural, fluent English)
  - "expected_refusal": true or false
  - "query_type": one of direct_content | semantic_synonym | meta_framing | off_topic
  - "vocabulary_note": required for semantic_synonym only — explain which words in the \
question bridge to different vocabulary in the source chunk

REQUIREMENTS (you must meet these exactly):
- 4 direct_content: the question uses key terms that appear LITERALLY in the chunks \
(baseline tests — should pass trivially)
- 5 semantic_synonym: the question expresses a concept from a chunk using DIFFERENT words \
that do NOT appear in the source chunk. The vocabulary gap must be genuine and non-trivial \
(not just plurals or minor morphological variants). vocabulary_note is required.
- 3 meta_framing: frame the question with phrases like "according to the video", "what does \
the speaker argue", "based on the content" — these test meta-phrase stripping in retrieval
- 4 off_topic: ask about things completely outside this playlist's domain \
(expected_refusal must be true for all four). Be clearly off-topic, not borderline.

Output ONLY the JSON array, no markdown, no explanation, no code fences."""


def _format_chunks(chunks: list) -> str:
    parts = []
    for i, doc in enumerate(chunks, 1):
        vid = doc.metadata.get("video_id", "?")
        title = doc.metadata.get("title", "Unknown")
        parts.append(f"[Chunk {i} | Video: {title} ({vid})]\n{doc.page_content[:600]}")
    return "\n\n---\n\n".join(parts)


def _sample_body_chunks(playlist_id: str) -> list:
    retriever = load_bm25(playlist_id)
    if retriever is None:
        raise ValueError(f"No BM25 index for playlist_id={playlist_id!r}. Ingest first.")

    body = [d for d in retriever.docs if d.metadata.get("chunk_type") == "body"]
    if not body:
        body = list(retriever.docs)

    # Group by video, take up to _CHUNKS_PER_VIDEO per video for diversity
    by_video: dict[str, list] = defaultdict(list)
    for doc in body:
        by_video[doc.metadata.get("video_id", "unknown")].append(doc)

    rng = random.Random(42)
    selected = []
    for vid_chunks in by_video.values():
        rng.shuffle(vid_chunks)
        selected.extend(vid_chunks[:_CHUNKS_PER_VIDEO])

    rng.shuffle(selected)
    return selected[:_MAX_CHUNKS]


def _golden_path(playlist_id: str) -> pathlib.Path:
    return _TESTS_DIR / f"golden_qa_{playlist_id}.json"


def generate_golden_dataset(playlist_id: str, user_id: str) -> list[dict]:
    """Generate and save a 16-example golden dataset for the given playlist.

    Calls Claude Opus with body chunks as context. Saves to
    tests/golden_qa_{playlist_id}.json. Returns the example list.

    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot generate golden dataset.")

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e

    chunks = _sample_body_chunks(playlist_id)
    prompt = _PROMPT_TEMPLATE.format(chunks=_format_chunks(chunks))

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    examples: list[dict] = json.loads(raw)
    for ex in examples:
        ex["playlist_id"] = playlist_id
        ex["user_id"] = user_id

    _TESTS_DIR.mkdir(parents=True, exist_ok=True)
    _golden_path(playlist_id).write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Golden dataset saved: {_golden_path(playlist_id)} ({len(examples)} examples)")
    return examples
