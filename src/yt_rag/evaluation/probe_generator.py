"""Tier 2 structural smoke probe generator.

Generates 5 on-topic probes from body chunks (Gemini Flash, same-vocabulary — tests
pipeline mechanics only, NOT vocabulary-gap retrieval) plus 3 hardcoded universal
off-topic probes. Saves to data/eval_probes/{playlist_id}.json and is called
automatically during ingestion.
"""
from __future__ import annotations

import json
import pathlib
import random

from langchain_core.documents import Document

from yt_rag.config import settings
from yt_rag.generation.chain import build_llm
from yt_rag.retrieval.bm25 import load_bm25

_PROBE_DIR = pathlib.Path("data/eval_probes")

_UNIVERSAL_OFF_TOPIC = [
    {
        "question": "In what year did Julius Caesar die?",
        "expected_refusal": True,
        "query_type": "universal_off_topic",
    },
    {
        "question": "What is the atomic mass of helium?",
        "expected_refusal": True,
        "query_type": "universal_off_topic",
    },
    {
        "question": "Who wrote Pride and Prejudice?",
        "expected_refusal": True,
        "query_type": "universal_off_topic",
    },
]


def _probe_path(playlist_id: str) -> pathlib.Path:
    return _PROBE_DIR / f"{playlist_id}.json"


def load_smoke_probes(playlist_id: str) -> list[dict] | None:
    """Return cached smoke probes, or None if not yet generated."""
    path = _probe_path(playlist_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_smoke_probes(playlist_id: str, user_id: str) -> list[dict]:
    """Generate and cache 5 on-topic + 3 off-topic structural probes.

    These test pipeline mechanics (does the pipeline run? do citations appear?).
    They do NOT test vocabulary-gap retrieval — questions are generated from the
    same chunks they'll retrieve, making them circular by design.
    """
    retriever = load_bm25(playlist_id)
    if retriever is None:
        raise ValueError(f"No BM25 index found for playlist_id={playlist_id!r}")

    body_chunks: list[Document] = [
        d for d in retriever.docs if d.metadata.get("chunk_type") == "body"
    ]
    if not body_chunks:
        body_chunks = retriever.docs  # fallback: use whatever is there

    rng = random.Random(42)
    samples = rng.sample(body_chunks, min(5, len(body_chunks)))

    llm = build_llm()
    on_topic: list[dict] = []
    for doc in samples:
        try:
            result = llm.invoke(
                "Write one specific factual question answerable only from this text. "
                "Output the question only, nothing else:\n\n" + doc.page_content[:800]
            )
            question = result.content.strip() if hasattr(result, "content") else str(result).strip()
            on_topic.append(
                {
                    "question": question,
                    "expected_refusal": False,
                    "query_type": "on_topic_smoke",
                    "playlist_id": playlist_id,
                    "user_id": user_id,
                }
            )
        except Exception:
            continue  # skip chunks where generation fails

    # Attach playlist/user context to off-topic probes too (needed by runner)
    off_topic = [
        {**p, "playlist_id": playlist_id, "user_id": user_id}
        for p in _UNIVERSAL_OFF_TOPIC
    ]

    probes = on_topic + off_topic
    _PROBE_DIR.mkdir(parents=True, exist_ok=True)
    _probe_path(playlist_id).write_text(
        json.dumps(probes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return probes
