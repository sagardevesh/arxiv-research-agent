import pytest

from rag.chunker import chunk_text


def test_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_produces_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    assert chunks[0].chunk_index == 0


def test_chunk_indices_are_sequential():
    text = "word " * 300  # ~1500 chars
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_overlap_causes_content_repetition():
    # With overlap, consecutive chunks should share some characters.
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # Each chunk should be non-empty.
    for c in chunks:
        assert c.text


def test_no_empty_chunks():
    text = "\n\n".join(["paragraph"] * 50)
    chunks = chunk_text(text, chunk_size=64, chunk_overlap=10)
    assert all(c.text for c in chunks)


def test_long_text_split_into_multiple_chunks():
    text = "sentence. " * 200
    chunks = chunk_text(text, chunk_size=128, chunk_overlap=16)
    assert len(chunks) > 1


def test_chunk_size_respected_approximately():
    text = "x" * 10000
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    # Hard character cut: no chunk should exceed chunk_size.
    for c in chunks:
        assert len(c.text) <= 500
