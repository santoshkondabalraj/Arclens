from __future__ import annotations

import re

from langchain_core.documents import Document

# YouTube noise tags: [Music], [Applause], [Laughter], etc.
_NOISE_TAG_RE = re.compile(r"\[[A-Z][a-zA-Z ]+\]")
# Timestamp sentinels inserted by loader: [T:123s]
_TIMESTAMP_SENTINEL_RE = re.compile(r"\[T:(\d+)s\]")
# Filler words — word-boundary match, case-insensitive
_FILLERS = (
    r"um|uh|hmm|uh-huh|mhm|you know|I mean|sort of|kind of|"
    r"right\?|right,|right\.|okay so|okay,|okay\.|so yeah|"
    r"\bum\b|\buh\b|\bhmm\b|\bmhm\b"
)
_FILLER_RE = re.compile(
    r"\b(?:um|uh|hmm|uh-huh|mhm|you know|I mean|sort of|kind of)\b",
    re.IGNORECASE,
)
# Multiple spaces
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
# Multi-newline runs → single paragraph break
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")


def _jaccard(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _dedup_rolling_overlap(text: str, window: int = 6, threshold: float = 0.8) -> str:
    """Remove auto-caption rolling overlap artifacts.

    YouTube auto-captions sometimes repeat the last N words of one line at the
    start of the next. Detect by Jaccard similarity of trailing/leading windows.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    cleaned: list[str] = []
    for i, sent in enumerate(sentences):
        if not sent.strip():
            continue
        if cleaned:
            prev_tokens = cleaned[-1].split()
            curr_tokens = sent.split()
            prev_tail = prev_tokens[-window:] if len(prev_tokens) >= window else prev_tokens
            curr_head = curr_tokens[:window] if len(curr_tokens) >= window else curr_tokens
            if _jaccard(prev_tail, curr_head) >= threshold:
                # Drop the duplicate prefix from curr
                overlap_len = len(curr_head)
                sent = " ".join(curr_tokens[overlap_len:])
        if sent.strip():
            cleaned.append(sent.strip())
    return " ".join(cleaned)


def clean_transcript(text: str) -> str:
    """Apply full cleaning pipeline to raw transcript text."""
    # 1. Strip timestamp sentinels ([T:Xs]) and YouTube noise tags
    text = _TIMESTAMP_SENTINEL_RE.sub("", text)
    text = _NOISE_TAG_RE.sub("", text)

    # 2. Remove filler words
    text = _FILLER_RE.sub("", text)

    # 3. Deduplicate rolling auto-caption overlap
    text = _dedup_rolling_overlap(text)

    # 4. Normalize multi-newline speaker-turn boundaries
    text = _MULTI_NEWLINE_RE.sub("\n", text)

    # 5. Collapse whitespace
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def clean_document(doc: Document) -> Document:
    """Return a new Document with cleaned page_content; metadata is preserved."""
    return Document(
        page_content=clean_transcript(doc.page_content),
        metadata=doc.metadata,
    )


def stamp_and_clean_document(doc: Document) -> Document:
    """Extract the first [T:Xs] sentinel as timestamp_seconds, then clean.

    Must be called on raw (uncleaned) chunks so the sentinels are still present.
    """
    m = _TIMESTAMP_SENTINEL_RE.search(doc.page_content)
    cleaned = Document(
        page_content=clean_transcript(doc.page_content),
        metadata={**doc.metadata},
    )
    if m:
        cleaned.metadata["timestamp_seconds"] = int(m.group(1))
    return cleaned
