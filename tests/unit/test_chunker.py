"""Unit tests for ingestion/chunker.py"""
import pytest
import tiktoken


def test_chunk_document_basic():
    from ingestion.chunker import chunk_document, ENCODING
    text = "Hello world. " * 200  # ~600 tokens
    chunks = chunk_document(text, doc_id="doc1", source_uri="test://doc")
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 512
        assert c.doc_id == "doc1"
        assert c.source_uri == "test://doc"


def test_chunk_hashes_unique():
    from ingestion.chunker import chunk_document
    text = "The quick brown fox jumps over the lazy dog. " * 100
    chunks = chunk_document(text, doc_id="doc1", source_uri="test://")
    hashes = [c.content_hash for c in chunks]
    assert len(set(hashes)) == len(hashes), "Chunk content hashes should be unique"


def test_chunk_overlap():
    from ingestion.chunker import chunk_document, ENCODING
    text = "word " * 600
    chunks = chunk_document(text, doc_id="d", source_uri="s", max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2
    tokens_0 = ENCODING.encode(chunks[0].content)
    tokens_1 = ENCODING.encode(chunks[1].content)
    overlap_text_0 = ENCODING.decode(tokens_0[-20:])
    assert overlap_text_0.strip() in chunks[1].content


def test_chunk_minhash_length():
    from ingestion.chunker import chunk_document, MINHASH_NUM_PERM
    text = "Some meaningful content for testing MinHash. " * 50
    chunks = chunk_document(text, doc_id="d", source_uri="s")
    if chunks and chunks[0].minhash_sig:
        assert len(chunks[0].minhash_sig) == MINHASH_NUM_PERM


def test_empty_content():
    from ingestion.chunker import chunk_document
    chunks = chunk_document("", doc_id="d", source_uri="s")
    assert chunks == []
