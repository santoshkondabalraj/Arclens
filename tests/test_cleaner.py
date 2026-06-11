from yt_rag.ingestion.cleaner import clean_transcript, clean_document
from langchain_core.documents import Document


def test_strips_timestamp_sentinels():
    text = "[T:0s] Hello [T:5s] world"
    assert "[T:" not in clean_transcript(text)


def test_strips_noise_tags():
    text = "Some speech [Music] more speech [Applause]"
    result = clean_transcript(text)
    assert "[Music]" not in result
    assert "[Applause]" not in result


def test_removes_filler_words():
    text = "So um this is like you know a test"
    result = clean_transcript(text)
    assert "um" not in result.split()


def test_collapses_whitespace():
    text = "hello   world"
    assert "  " not in clean_transcript(text)


def test_clean_document_preserves_metadata():
    doc = Document(
        page_content="[T:0s] hello um world",
        metadata={"video_id": "abc", "title": "Test"},
    )
    cleaned = clean_document(doc)
    assert cleaned.metadata["video_id"] == "abc"
    assert "[T:" not in cleaned.page_content
