from __future__ import annotations

import re
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

try:
    from pytubefix import Playlist, YouTube
    _PYTUBE_AVAILABLE = True
except Exception:
    _PYTUBE_AVAILABLE = False


def _extract_playlist_video_ids(playlist_url: str) -> list[str]:
    """Return ordered list of video IDs from a YouTube playlist URL."""
    if _PYTUBE_AVAILABLE:
        pl = Playlist(playlist_url)
        return list(pl.video_urls)
    # Fallback: parse list= param and use YouTube Data API conventions
    import urllib.parse
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(playlist_url).query)
    raise ValueError(
        f"pytube not available; cannot enumerate playlist {playlist_url}. "
        "Install pytube or pass individual video URLs."
    )


def _video_id_from_url(url: str) -> str:
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    # Short URL: youtu.be/<id>
    return parsed.path.lstrip("/")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)
def _fetch_transcript(video_id: str, language: str = "en") -> list[dict]:
    # v1.x API: instantiate then call fetch()
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
    return [{"text": s.text, "start": s.start, "duration": s.duration} for s in transcript]


def _get_video_metadata(video_url: str) -> dict:
    """Return title, channel, publish_date for a video URL via pytube."""
    if not _PYTUBE_AVAILABLE:
        video_id = _video_id_from_url(video_url)
        return {"title": video_id, "channel": "unknown", "publish_date": "unknown"}
    yt = YouTube(video_url)
    return {
        "title": yt.title or "Unknown Title",
        "channel": yt.author or "Unknown Channel",
        "publish_date": str(yt.publish_date.date()) if yt.publish_date else "unknown",
    }


class YouTubePlaylistLoader(BaseLoader):
    """Load transcripts from a YouTube playlist as LangChain Documents.

    Each video produces one Document with its full transcript.
    Timestamps are embedded as [T:{seconds}s] sentinels to survive downstream
    cleaning steps and be available for citation extraction.
    """

    def __init__(self, playlist_url: str, language: str = "en") -> None:
        self.playlist_url = playlist_url
        self.language = language

    def lazy_load(self) -> Iterator[Document]:
        video_urls = _extract_playlist_video_ids(self.playlist_url)
        for video_url in video_urls:
            video_id = _video_id_from_url(video_url)
            try:
                entries = _fetch_transcript(video_id, self.language)
            except (TranscriptsDisabled, NoTranscriptFound):
                continue

            meta = _get_video_metadata(video_url)
            # Concatenate transcript, embedding sentinel timestamps every entry
            parts = [
                f"[T:{int(e['start'])}s] {e['text']}" for e in entries
            ]
            page_content = " ".join(parts)

            yield Document(
                page_content=page_content,
                metadata={
                    "video_id": video_id,
                    "video_url": video_url,
                    "title": meta["title"],
                    "channel": meta["channel"],
                    "publish_date": meta["publish_date"],
                    "playlist_url": self.playlist_url,
                    "chunk_type": "raw_transcript",
                },
            )

    def load_metadata_chunks(self) -> list[Document]:
        """Return one short Document per video containing title + description metadata."""
        docs = []
        video_urls = _extract_playlist_video_ids(self.playlist_url)
        for video_url in video_urls:
            video_id = _video_id_from_url(video_url)
            meta = _get_video_metadata(video_url)
            content = f"Title: {meta['title']}\nChannel: {meta['channel']}\nPublished: {meta['publish_date']}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "video_id": video_id,
                        "video_url": video_url,
                        "title": meta["title"],
                        "channel": meta["channel"],
                        "publish_date": meta["publish_date"],
                        "playlist_url": self.playlist_url,
                        "chunk_type": "metadata",
                    },
                )
            )
        return docs
