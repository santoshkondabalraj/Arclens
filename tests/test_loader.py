"""Unit tests for YouTubePlaylistLoader.

These tests mock youtube-transcript-api and pytube to avoid network calls.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_transcript():
    return [
        {"text": "Hello world", "start": 0.0, "duration": 2.0},
        {"text": "This is a test", "start": 2.5, "duration": 3.0},
    ]


@pytest.fixture()
def mock_video_meta():
    m = MagicMock()
    m.title = "Test Episode"
    m.author = "Test Channel"
    m.publish_date.date.return_value = "2025-01-01"
    return m


def test_loader_embeds_sentinel_timestamps(mock_transcript, mock_video_meta):
    with (
        patch("yt_rag.ingestion.loader._extract_playlist_video_ids", return_value=["https://youtube.com/watch?v=abc123"]),
        patch("yt_rag.ingestion.loader._fetch_transcript", return_value=mock_transcript),
        patch("yt_rag.ingestion.loader.YouTube", return_value=mock_video_meta),
        patch("yt_rag.ingestion.loader._PYTUBE_AVAILABLE", True),
    ):
        from yt_rag.ingestion.loader import YouTubePlaylistLoader
        loader = YouTubePlaylistLoader("https://youtube.com/playlist?list=TEST")
        docs = list(loader.lazy_load())

    assert len(docs) == 1
    assert "[T:0s]" in docs[0].page_content
    assert docs[0].metadata["chunk_type"] == "raw_transcript"
    assert docs[0].metadata["video_id"] == "abc123"


def test_loader_metadata_chunk(mock_video_meta):
    with (
        patch("yt_rag.ingestion.loader._extract_playlist_video_ids", return_value=["https://youtube.com/watch?v=abc123"]),
        patch("yt_rag.ingestion.loader.YouTube", return_value=mock_video_meta),
        patch("yt_rag.ingestion.loader._PYTUBE_AVAILABLE", True),
    ):
        from yt_rag.ingestion.loader import YouTubePlaylistLoader
        loader = YouTubePlaylistLoader("https://youtube.com/playlist?list=TEST")
        docs = loader.load_metadata_chunks()

    assert len(docs) == 1
    assert docs[0].metadata["chunk_type"] == "metadata"
    assert "Test Episode" in docs[0].page_content
