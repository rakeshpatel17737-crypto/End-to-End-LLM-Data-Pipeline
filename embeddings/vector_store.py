"""pgvector operations: upsert chunks and MMR retrieval."""
from __future__ import annotations

import json
import uuid

import numpy as np
import psycopg2
import psycopg2.extras


def upsert_chunk(conn, chunk_data: dict) -> None:
    """Insert or update a chunk row including its embedding."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks
                (chunk_id, doc_id, chunk_index, content, content_hash,
                 embedding, token_count, source_uri, page_number,
                 language, quality_score, has_pii, minhash_sig, embedded_at)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                content        = EXCLUDED.content,
                content_hash   = EXCLUDED.content_hash,
                embedding      = EXCLUDED.embedding,
                token_count    = EXCLUDED.token_count,
                quality_score  = EXCLUDED.quality_score,
                has_pii        = EXCLUDED.has_pii,
                minhash_sig    = EXCLUDED.minhash_sig,
                embedded_at    = NOW()
            """,
            [
                chunk_data.get("chunk_id") or str(uuid.uuid4()),
                chunk_data["doc_id"],
                chunk_data["chunk_index"],
                chunk_data["content"],
                chunk_data["content_hash"],
                chunk_data["embedding"],
                chunk_data["token_count"],
                chunk_data["source_uri"],
                chunk_data.get("page_number"),
                chunk_data.get("language"),
                chunk_data.get("quality_score"),
                chunk_data.get("has_pii", False),
                json.dumps(chunk_data.get("minhash_sig", [])),
            ],
        )
    conn.commit()


def upsert_chunks_batch(conn, chunks: list[dict]) -> None:
    """Bulk upsert a list of chunk dicts."""
    for chunk in chunks:
        upsert_chunk(conn, chunk)


def search_mmr(
    conn,
    query_embedding: list[float],
    top_k: int = 5,
    mmr_lambda: float = 0.5,
    fetch_k: int = 50,
) -> tuple[list[dict], float]:
    """MMR retrieval: fetch_k by cosine sim, then greedily select top_k for diversity.

    Returns (selected_chunks, mmr_diversity_score).
    mmr_diversity = 1 - mean_pairwise_cosine(selected_chunks)
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, content, source_uri, page_number,
                   quality_score, language,
                   1 - (embedding <=> %s::vector) AS cosine_sim,
                   embedding
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [query_embedding, query_embedding, fetch_k],
        )
        candidates = [dict(r) for r in cur.fetchall()]

    if not candidates:
        return [], 0.0

    if len(candidates) <= top_k:
        selected = candidates
        diversity = _mmr_diversity([c["embedding"] for c in selected])
        for c in selected:
            c.pop("embedding", None)
        return selected, diversity

    query_vec = np.array(query_embedding, dtype=np.float32)
    embs = np.array([c["embedding"] for c in candidates], dtype=np.float32)

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(top_k):
        best_idx = None
        best_score = float("-inf")
        for i in remaining:
            sim_to_query = float(
                np.dot(embs[i], query_vec) /
                (np.linalg.norm(embs[i]) * np.linalg.norm(query_vec) + 1e-9)
            )
            if selected_indices:
                max_sim_to_selected = max(
                    float(np.dot(embs[i], embs[j]) /
                          (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-9))
                    for j in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            score = mmr_lambda * sim_to_query - (1 - mmr_lambda) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    selected = [candidates[i] for i in selected_indices]
    diversity = _mmr_diversity([embs[i].tolist() for i in selected_indices])

    for c in selected:
        c.pop("embedding", None)

    return selected, diversity


def _mmr_diversity(embeddings: list) -> float:
    if len(embeddings) < 2:
        return 1.0
    arr = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    normalized = arr / norms
    sim_matrix = normalized @ normalized.T
    n = len(embeddings)
    off_diag = sim_matrix[np.triu_indices(n, k=1)]
    return float(1.0 - off_diag.mean())


def _to_list(vec) -> list[float]:
    """pgvector returns numpy arrays; convert to native Python list for JSON/XCom."""
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return [float(x) for x in vec]


def fetch_embedding_sample(conn, n: int = 500, age_days_max: int = 7) -> list[list[float]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT embedding FROM chunks
            WHERE embedding IS NOT NULL
              AND created_at >= NOW() - INTERVAL '%s days'
            ORDER BY RANDOM()
            LIMIT %s
            """,
            [age_days_max, n],
        )
        return [_to_list(row[0]) for row in cur.fetchall()]


def fetch_baseline_embeddings(conn, n: int = 500, age_days_min: int = 30) -> list[list[float]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT embedding FROM chunks
            WHERE embedding IS NOT NULL
              AND created_at < NOW() - INTERVAL '%s days'
            ORDER BY RANDOM()
            LIMIT %s
            """,
            [age_days_min, n],
        )
        return [_to_list(row[0]) for row in cur.fetchall()]
