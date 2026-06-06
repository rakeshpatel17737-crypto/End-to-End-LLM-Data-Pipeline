"""Airflow DAG: document ingestion pipeline.

Task graph:
  load_document_sources
    → check_new_sources [Branch]
        → skip_no_sources
        → fetch_raw_content
            → chunk_text
                → embed_chunks
                    → store_embeddings
                        → log_pipeline_run

XCom: each task pushes results; log_pipeline_run aggregates all for pipeline_runs row.
DLQ: on_failure_callback inserts into dead_letter_queue on permanent failure.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.insert(0, "/opt/llm_platform")

DEFAULT_ARGS = {
    "owner": "llm-platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "email_on_failure": False,
}

DOCUMENT_SOURCES = [
    {"uri": "https://en.wikipedia.org/wiki/Apache_Airflow", "source_type": "web"},
    {"uri": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation", "source_type": "web"},
    {"uri": "https://en.wikipedia.org/wiki/Large_language_model", "source_type": "web"},
]


def _pg_conn():
    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )
    register_vector(conn)
    return conn


def _redis_client():
    import redis
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "redis"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=False,
    )


def _dlq_insert(conn, source_uri: str, source_type: str, failure_reason: str, failure_stage: str):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dead_letter_queue
                    (source_uri, source_type, failure_reason, failure_stage)
                VALUES (%s, %s, %s, %s)
                """,
                [source_uri, source_type, failure_reason[:500], failure_stage],
            )
        conn.commit()
    except Exception:
        conn.rollback()


def load_document_sources(ti, **kwargs):
    conf = kwargs.get("dag_run", {})
    run_conf = getattr(conf, "conf", {}) or {}
    sources = run_conf.get("sources", DOCUMENT_SOURCES)
    ti.xcom_push(key="sources", value=sources)
    ti.xcom_push(key="run_id", value=str(uuid.uuid4()))
    print(f"Loaded {len(sources)} document sources")


def check_new_sources(ti, **kwargs):
    sources = ti.xcom_pull(task_ids="load_document_sources", key="sources")
    if not sources:
        return "skip_no_sources"
    conn = _pg_conn()
    try:
        pending = []
        for src in sources:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ingest_status FROM documents WHERE source_uri = %s",
                    [src["uri"]],
                )
                row = cur.fetchone()
                if row is None or row[0] in ("pending", "failed"):
                    pending.append(src)
        conn.close()
        if not pending:
            print("All sources already ingested")
            return "skip_no_sources"
        ti.xcom_push(key="pending_sources", value=pending)
        return "fetch_raw_content"
    except Exception:
        conn.close()
        return "skip_no_sources"


def fetch_raw_content(ti, **kwargs):
    from ingestion.loaders.web_loader import load_web
    from ingestion.loaders.pdf_loader import load_pdf
    from ingestion.loaders.api_loader import load_api

    sources = ti.xcom_pull(task_ids="check_new_sources", key="pending_sources") or \
              ti.xcom_pull(task_ids="load_document_sources", key="sources")
    conn = _pg_conn()
    raw_docs = []

    for src in sources:
        uri = src["uri"]
        stype = src.get("source_type", "web")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (source_uri, source_type, ingest_status)
                    VALUES (%s, %s, 'processing')
                    ON CONFLICT (source_uri) DO UPDATE SET ingest_status = 'processing', updated_at = NOW()
                    RETURNING doc_id
                    """,
                    [uri, stype],
                )
                doc_id = str(cur.fetchone()[0])
            conn.commit()

            if stype == "web":
                doc = load_web(uri)
            elif stype == "pdf":
                doc = load_pdf(uri)
            else:
                doc = load_api(uri)

            raw_docs.append({
                "doc_id": doc_id,
                "source_uri": uri,
                "source_type": stype,
                "title": doc.title,
                "pages": doc.pages,
                "total_chars": doc.total_chars,
            })
        except Exception as exc:
            print(f"Failed to fetch {uri}: {exc}")
            _dlq_insert(conn, uri, stype, str(exc), "load")
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE documents SET ingest_status='failed' WHERE source_uri=%s", [uri]
                    )
                conn.commit()
            except Exception:
                pass

    conn.close()
    ti.xcom_push(key="raw_docs", value=raw_docs)
    print(f"Fetched {len(raw_docs)}/{len(sources)} documents")


def chunk_text(ti, **kwargs):
    from ingestion.chunker import chunk_document, Chunk
    import dataclasses

    raw_docs = ti.xcom_pull(task_ids="fetch_raw_content", key="raw_docs") or []
    all_chunks = []

    for doc in raw_docs:
        full_text = "\n\n".join(doc["pages"])
        chunks = []
        for page_num, page_text in enumerate(doc["pages"]):
            page_chunks = chunk_document(
                content=page_text,
                doc_id=doc["doc_id"],
                source_uri=doc["source_uri"],
                page_number=page_num if len(doc["pages"]) > 1 else None,
            )
            chunks.extend(page_chunks)

        all_chunks.append({
            "doc_id": doc["doc_id"],
            "source_uri": doc["source_uri"],
            "title": doc.get("title"),
            "chunks": [dataclasses.asdict(c) for c in chunks],
        })

    ti.xcom_push(key="chunked_docs", value=all_chunks)
    total = sum(len(d["chunks"]) for d in all_chunks)
    print(f"Created {total} chunks from {len(all_chunks)} documents")


def embed_chunks(ti, **kwargs):
    import redis
    from embeddings.embedding_cache import EmbeddingCache
    from embeddings.embedder import embed_batch
    from warehouse.llm_logger import LLMLogger

    chunked_docs = ti.xcom_pull(task_ids="chunk_text", key="chunked_docs") or []
    run_id = ti.xcom_pull(task_ids="load_document_sources", key="run_id")

    r = _redis_client()
    cache = EmbeddingCache(r)
    llm_logger = LLMLogger(os.environ.get("DUCKDB_PATH", "/data/warehouse/llm_logs.duckdb"))

    embedded_docs = []
    for doc in chunked_docs:
        chunks = doc["chunks"]
        if not chunks:
            continue

        min_quality = float(os.environ.get("MIN_CHUNK_QUALITY", "0.3"))
        good_chunks = [c for c in chunks if c.get("quality_score", 0) >= min_quality]
        print(f"Doc {doc['doc_id']}: {len(good_chunks)}/{len(chunks)} chunks passed quality filter")

        texts = [c["content"] for c in good_chunks]
        chunk_ids = [c.get("chunk_id", str(uuid.uuid4())) for c in good_chunks]

        results = embed_batch(texts, chunk_ids, cache, llm_logger, run_id or "airflow")

        for chunk, result in zip(good_chunks, results):
            chunk["embedding"] = result.embedding
            chunk["cost_usd"] = result.cost_usd
            chunk["from_cache"] = result.from_cache

        total_cost = sum(r.cost_usd for r in results)
        cache_hits = sum(1 for r in results if r.from_cache)

        embedded_docs.append({
            **doc,
            "chunks": good_chunks,
            "total_cost_usd": total_cost,
            "cache_hits": cache_hits,
            "tokens_embedded": sum(c.get("token_count", 0) for c in good_chunks),
        })

    ti.xcom_push(key="embedded_docs", value=embedded_docs)
    total_cost = sum(d["total_cost_usd"] for d in embedded_docs)
    print(f"Embedded {sum(len(d['chunks']) for d in embedded_docs)} chunks, cost=${total_cost:.6f}")


def store_embeddings(ti, **kwargs):
    from embeddings.vector_store import upsert_chunk

    embedded_docs = ti.xcom_pull(task_ids="embed_chunks", key="embedded_docs") or []
    conn = _pg_conn()

    for doc in embedded_docs:
        doc_id = doc["doc_id"]
        chunks = doc["chunks"]
        chunk_count = len(chunks)
        total_tokens = sum(c.get("token_count", 0) for c in chunks)
        total_cost = doc.get("total_cost_usd", 0.0)

        for chunk in chunks:
            try:
                upsert_chunk(conn, chunk)
            except Exception as exc:
                print(f"Failed to store chunk {chunk.get('chunk_index')}: {exc}")
                conn.rollback()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents SET
                        ingest_status = 'done',
                        chunk_count = %s,
                        total_tokens = %s,
                        ingest_cost_usd = %s,
                        ingested_at = NOW(),
                        updated_at = NOW()
                    WHERE doc_id = %s
                    """,
                    [chunk_count, total_tokens, total_cost, doc_id],
                )
            conn.commit()
            print(f"Stored {chunk_count} chunks for doc {doc_id}")
        except Exception as exc:
            conn.rollback()
            print(f"Failed to update doc {doc_id}: {exc}")

    conn.close()
    ti.xcom_push(key="docs_succeeded", value=len(embedded_docs))


def log_pipeline_run(ti, **kwargs):
    import time

    run_id = ti.xcom_pull(task_ids="load_document_sources", key="run_id")
    sources = ti.xcom_pull(task_ids="load_document_sources", key="sources") or []
    embedded_docs = ti.xcom_pull(task_ids="embed_chunks", key="embedded_docs") or []
    docs_succeeded = ti.xcom_pull(task_ids="store_embeddings", key="docs_succeeded") or 0

    chunks_created = sum(len(d["chunks"]) for d in embedded_docs)
    tokens_embedded = sum(d.get("tokens_embedded", 0) for d in embedded_docs)
    embed_cost = sum(d.get("total_cost_usd", 0.0) for d in embedded_docs)

    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_runs
                    (dag_id, airflow_run_id, status, docs_attempted, docs_succeeded,
                     docs_failed, chunks_created, tokens_embedded, embed_cost_usd, completed_at)
                VALUES (%s, %s, 'success', %s, %s, %s, %s, %s, %s, NOW())
                """,
                [
                    "ingest_documents_dag",
                    ti.run_id,
                    len(sources),
                    docs_succeeded,
                    len(sources) - docs_succeeded,
                    chunks_created,
                    tokens_embedded,
                    embed_cost,
                ],
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        print(f"Failed to log pipeline run: {exc}")
    finally:
        conn.close()

    print(f"Pipeline run logged: {docs_succeeded} docs, {chunks_created} chunks, cost=${embed_cost:.6f}")


with DAG(
    dag_id="ingest_documents_dag",
    default_args=DEFAULT_ARGS,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["llm-platform", "ingestion"],
) as dag:

    t_load = PythonOperator(task_id="load_document_sources", python_callable=load_document_sources)
    t_check = BranchPythonOperator(task_id="check_new_sources", python_callable=check_new_sources)
    t_skip = EmptyOperator(task_id="skip_no_sources")
    t_fetch = PythonOperator(task_id="fetch_raw_content", python_callable=fetch_raw_content)
    t_chunk = PythonOperator(task_id="chunk_text", python_callable=chunk_text)
    t_embed = PythonOperator(task_id="embed_chunks", python_callable=embed_chunks)
    t_store = PythonOperator(task_id="store_embeddings", python_callable=store_embeddings)
    t_log = PythonOperator(task_id="log_pipeline_run", python_callable=log_pipeline_run)

    t_load >> t_check >> [t_skip, t_fetch]
    t_fetch >> t_chunk >> t_embed >> t_store >> t_log
