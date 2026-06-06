"""OpenAI batch embedder with Redis cache and DuckDB cost tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass

import openai

from .config import config
from .embedding_cache import EmbeddingCache


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
    client = openai.OpenAI(api_key=config.openai_api_key)
    batch_size = config.max_batch_size

    for batch_start in range(0, len(miss_indices), batch_size):
        batch_idx = miss_indices[batch_start: batch_start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        batch_chunk_ids = [chunk_ids[i] for i in batch_idx]

        t0 = time.perf_counter()
        response = client.embeddings.create(
            model=config.embedding_model,
            input=batch_texts,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        tokens_used = response.usage.total_tokens
        cost_usd = _cost(tokens_used)

        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        cache.mset(batch_texts, embeddings)

        llm_logger.log(
            request_id=request_id,
            interaction_type="embedding",
            model=config.embedding_model,
            tokens_in=tokens_used,
            tokens_out=0,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            status="success",
            metadata={"batch_size": len(batch_texts), "chunk_ids": batch_chunk_ids},
        )

        per_item_cost = cost_usd / len(batch_texts)
        per_item_latency = latency_ms / len(batch_texts)

        for local_i, (global_i, chunk_id, embedding) in enumerate(
            zip(batch_idx, batch_chunk_ids, embeddings)
        ):
            results[global_i] = EmbeddingResult(
                chunk_id=chunk_id,
                embedding=embedding,
                tokens_used=tokens_used // len(batch_texts),
                cost_usd=per_item_cost,
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


def _cost(tokens: int) -> float:
    return (tokens / 1000.0) * config.price_per_1k_tokens
