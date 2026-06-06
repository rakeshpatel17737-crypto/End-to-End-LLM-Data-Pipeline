"""Local embedder using sentence-transformers, with Redis cache and DuckDB logging.

Embeddings run locally on CPU — no API key, zero cost. Token counts are approximated
for the cost/usage dashboard; cost_usd is always 0.0 for local embeddings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .config import config
from .embedding_cache import EmbeddingCache

_model = None


def _get_model():
    """Lazy-load the sentence-transformers model (downloaded once, cached on disk)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.embedding_model)
    return _model


@dataclass
class EmbeddingResult:
    chunk_id: str
    embedding: list[float]
    tokens_used: int
    cost_usd: float
    latency_ms: float
    from_cache: bool


def embed_batch(
    texts: list[str],
    chunk_ids: list[str],
    cache: EmbeddingCache,
    llm_logger,          # warehouse.llm_logger.LLMLogger
    request_id: str,
) -> list[EmbeddingResult]:
    assert len(texts) == len(chunk_ids)

    cached = cache.mget(texts)
    results: list[EmbeddingResult | None] = [None] * len(texts)
    miss_indices: list[int] = []

    for i, (cached_emb, chunk_id) in enumerate(zip(cached, chunk_ids)):
        if cached_emb is not None:
            results[i] = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=cached_emb,
                tokens_used=0,
                cost_usd=0.0,
                latency_ms=0.0,
                from_cache=True,
            )
            llm_logger.log(
                request_id=request_id,
                interaction_type="embedding",
                model=config.embedding_model,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                latency_ms=0.0,
                status="cached",
                metadata={"chunk_id": chunk_id},
            )
        else:
            miss_indices.append(i)

    if miss_indices:
        _embed_misses(texts, chunk_ids, miss_indices, cache, llm_logger, request_id, results)

    return results  # type: ignore[return-value]


def _embed_misses(
    texts: list[str],
    chunk_ids: list[str],
    miss_indices: list[int],
    cache: EmbeddingCache,
    llm_logger,
    request_id: str,
    results: list,
) -> None:
    model = _get_model()
    miss_texts = [texts[i] for i in miss_indices]

    t0 = time.perf_counter()
    embeddings = model.encode(
        miss_texts,
        batch_size=config.max_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    embeddings_list = [e.tolist() for e in embeddings]
    cache.mset(miss_texts, embeddings_list)

    total_tokens = sum(_approx_tokens(t) for t in miss_texts)
    llm_logger.log(
        request_id=request_id,
        interaction_type="embedding",
        model=config.embedding_model,
        tokens_in=total_tokens,
        tokens_out=0,
        cost_usd=0.0,        # local embeddings are free
        latency_ms=latency_ms,
        status="success",
        metadata={"batch_size": len(miss_texts)},
    )

    per_item_latency = latency_ms / len(miss_texts)
    for local_i, global_i in enumerate(miss_indices):
        results[global_i] = EmbeddingResult(
            chunk_id=chunk_ids[global_i],
            embedding=embeddings_list[local_i],
            tokens_used=_approx_tokens(texts[global_i]),
            cost_usd=0.0,
            latency_ms=per_item_latency,
            from_cache=False,
        )


def embed_single(
    text: str,
    cache: EmbeddingCache,
    llm_logger,
    request_id: str,
    chunk_id: str = "query",
) -> EmbeddingResult:
    results = embed_batch([text], [chunk_id], cache, llm_logger, request_id)
    return results[0]


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for usage tracking."""
    return max(1, len(text) // 4)
