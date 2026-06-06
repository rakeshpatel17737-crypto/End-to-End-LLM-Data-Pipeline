"""Token-aware text chunker with metadata tagging."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")
MINHASH_NUM_PERM = 128


@dataclass
class Chunk:
    content: str
    chunk_index: int
    doc_id: str
    source_uri: str
    page_number: int | None
    token_count: int
    content_hash: str
    language: str | None
    quality_score: float
    has_pii: bool
    minhash_sig: list[int] = field(default_factory=list)


def chunk_document(
    content: str,
    doc_id: str,
    source_uri: str,
    page_number: int | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Sliding-window token-aware chunker with SHA-256 + MinHash fingerprinting."""
    from ingestion.quality_scorer import score_chunk

    tokens = ENCODING.encode(content)
    if not tokens:
        return []

    step = max_tokens - overlap_tokens
    chunks: list[Chunk] = []
    idx = 0
    chunk_index = 0

    while idx < len(tokens):
        window = tokens[idx: idx + max_tokens]
        chunk_text = ENCODING.decode(window)

        content_hash = _sha256(chunk_text)
        minhash = _minhash(chunk_text)
        report = score_chunk(chunk_text, content_hash, set())

        chunk = Chunk(
            content=chunk_text,
            chunk_index=chunk_index,
            doc_id=doc_id,
            source_uri=source_uri,
            page_number=page_number,
            token_count=len(window),
            content_hash=content_hash,
            language=report.language,
            quality_score=report.quality_score,
            has_pii=report.has_pii,
            minhash_sig=minhash,
        )
        chunks.append(chunk)
        idx += step
        chunk_index += 1

    return chunks


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _minhash(text: str) -> list[int]:
    try:
        from datasketch import MinHash
        m = MinHash(num_perm=MINHASH_NUM_PERM)
        for word in text.lower().split():
            m.update(word.encode("utf-8"))
        return list(m.hashvalues.tolist())
    except ImportError:
        return []
