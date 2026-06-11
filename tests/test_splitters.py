from langchain_core.documents import Document

from yt_rag.chunking.splitters import split_documents, _token_len


def _make_doc(content: str, chunk_type: str = "raw_transcript") -> Document:
    return Document(
        page_content=content,
        metadata={
            "video_id": "vid1",
            "title": "Test Video",
            "channel": "Test Channel",
            "chunk_type": chunk_type,
        },
    )


def test_body_chunks_have_correct_type():
    doc = _make_doc("word " * 800)
    chunks = split_documents([doc])
    body = [c for c in chunks if c.metadata["chunk_type"] == "body"]
    assert len(body) > 0


def test_metadata_chunks_have_correct_type():
    doc = _make_doc("Title: Foo\nChannel: Bar", chunk_type="metadata")
    chunks = split_documents([doc])
    meta = [c for c in chunks if c.metadata["chunk_type"] == "metadata"]
    assert len(meta) > 0


def test_chunk_index_is_sequential():
    doc = _make_doc("word " * 1500)
    chunks = split_documents([doc])
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_token_count_populated():
    doc = _make_doc("hello world this is a test")
    chunks = split_documents([doc])
    for c in chunks:
        assert "token_count" in c.metadata
        assert c.metadata["token_count"] > 0


def test_body_chunk_size_within_bounds():
    doc = _make_doc("word " * 2000)
    chunks = split_documents([doc])
    body = [c for c in chunks if c.metadata["chunk_type"] == "body"]
    for c in body:
        # Allow slight overshoot from splitter heuristics
        assert _token_len(c.page_content) <= 700
