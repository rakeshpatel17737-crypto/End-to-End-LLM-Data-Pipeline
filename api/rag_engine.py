"""RAG query pipeline: embed → MMR retrieve → Claude generate → log."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import anthropic
import psycopg2

from embeddings.embedder import embed_single
from embeddings.embedding_cache import EmbeddingCache
from embeddings.vector_store import search_mmr
from warehouse.llm_logger import LLMLogger

RAG_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context chunks.
If the context does not contain enough information to answer confidently, say so clearly.
Be concise. For each fact you state, cite the source_uri of the chunk it came from."""

RAG_SYSTEM_PROMPT_TOKENS_APPROX = 60


@dataclass
class RAGResponse:
    answer: str
    source_chunks: list[dict]
    mmr_diversity: float
    mean_cosine_sim: float
    latency_ms: float
    cost_usd: float
    request_id: str


async def run_rag_query(
    question: str,
    pg_conn,
    cache: EmbeddingCache,
    llm_logger: LLMLogger,
    anthropic_client: anthropic.AsyncAnthropic,
    anthropic_model: str,
    top_k: int = 5,
    mmr_lambda: float = 0.5,
    request_id: str | None = None,
) -> RAGResponse:
    request_id = request_id or str(uuid.uuid4())
    t_total = time.perf_counter()

    query_result = embed_single(
        text=question,
        cache=cache,
        llm_logger=llm_logger,
        request_id=request_id,
        chunk_id="rag_query",
    )

    chunks, mmr_diversity = search_mmr(
        pg_conn, query_result.embedding, top_k=top_k, mmr_lambda=mmr_lambda
    )

    mean_cosine_sim = (
        float(sum(c.get("cosine_sim", 0) or 0 for c in chunks) / len(chunks))
        if chunks else 0.0
    )

    _log_retrieval_metrics(
        pg_conn,
        request_id=request_id,
        query_text=question,
        top_k=top_k,
        mmr_diversity=mmr_diversity,
        mean_cosine_sim=mean_cosine_sim,
        result_count=len(chunks),
        latency_ms=(time.perf_counter() - t_total) * 1000,
    )

    if not chunks:
        return RAGResponse(
            answer="I could not find relevant information in the knowledge base.",
            source_chunks=[],
            mmr_diversity=0.0,
            mean_cosine_sim=0.0,
            latency_ms=(time.perf_counter() - t_total) * 1000,
            cost_usd=query_result.cost_usd,
            request_id=request_id,
        )

    context = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    t_llm = time.perf_counter()
    response = await anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        system=[{
            "type": "text",
            "text": RAG_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_message}],
    )
    llm_latency_ms = (time.perf_counter() - t_llm) * 1000

    answer = response.content[0].text if response.content else ""
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    rag_cost = _claude_cost(tokens_in, tokens_out, anthropic_model)

    llm_logger.log(
        request_id=request_id,
        interaction_type="rag_query",
        model=anthropic_model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=rag_cost,
        latency_ms=llm_latency_ms,
        status="success",
        prompt_text=user_message[:2000],
        response_text=answer[:2000],
        metadata={"top_k": top_k, "mmr_diversity": round(mmr_diversity, 4)},
    )

    total_cost = query_result.cost_usd + rag_cost
    total_latency = (time.perf_counter() - t_total) * 1000

    return RAGResponse(
        answer=answer,
        source_chunks=chunks,
        mmr_diversity=round(mmr_diversity, 4),
        mean_cosine_sim=round(mean_cosine_sim, 4),
        latency_ms=round(total_latency, 2),
        cost_usd=round(total_cost, 8),
        request_id=request_id,
    )


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {c.get('source_uri', 'unknown')}\n{c.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def _log_retrieval_metrics(
    conn, *, request_id, query_text, top_k, mmr_diversity, mean_cosine_sim, result_count, latency_ms
) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO retrieval_metrics
                    (request_id, query_text, top_k, mmr_diversity, mean_cosine_sim, result_count, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [request_id, query_text[:500], top_k,
                 round(mmr_diversity, 4), round(mean_cosine_sim, 4), result_count, round(latency_ms, 2)],
            )
        conn.commit()
    except Exception:
        conn.rollback()


def _claude_cost(tokens_in: int, tokens_out: int, model: str) -> float:
    if "sonnet" in model:
        return (tokens_in / 1_000_000) * 3.0 + (tokens_out / 1_000_000) * 15.0
    return (tokens_in / 1_000_000) * 1.0 + (tokens_out / 1_000_000) * 5.0
