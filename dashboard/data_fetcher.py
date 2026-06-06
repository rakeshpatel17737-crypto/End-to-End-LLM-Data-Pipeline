"""Data fetching utilities for the Streamlit dashboard.

Adapted from feature-store/monitoring/data_fetcher.py — same singleton engine + Redis pattern,
plus DuckDB as a third data source for LLM interaction logs.
"""
from __future__ import annotations

import os
from typing import Optional

import duckdb
import pandas as pd
import redis as redis_lib
import requests
from sqlalchemy import create_engine, text

_redis_client: Optional[redis_lib.Redis] = None
_db_engine = None

API_URL = os.getenv("API_URL", "http://localhost:8000")
POSTGRES_DSN = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER','llmplatform')}:"
    f"{os.getenv('POSTGRES_PASSWORD','llmplatform')}@"
    f"{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT','5432')}/"
    f"{os.getenv('POSTGRES_DB','llm_platform')}"
)
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/data/warehouse/llm_logs.duckdb")


def get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_connect_timeout=3,
        )
    return _redis_client


def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(POSTGRES_DSN, pool_pre_ping=True)
    return _db_engine


def duckdb_query(sql: str) -> pd.DataFrame:
    try:
        with duckdb.connect(DUCKDB_PATH, read_only=True) as con:
            return con.execute(sql).df()
    except Exception:
        return pd.DataFrame()


def fetch_api_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "db": "unknown", "redis": "unknown", "duckdb": "unknown"}


def fetch_pipeline_runs(limit: int = 30) -> pd.DataFrame:
    try:
        with get_db_engine().connect() as conn:
            return pd.read_sql(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT %(limit)s",
                conn,
                params={"limit": limit},
            )
    except Exception:
        return pd.DataFrame()


def fetch_dlq_pending() -> pd.DataFrame:
    try:
        with get_db_engine().connect() as conn:
            return pd.read_sql(
                "SELECT source_uri, source_type, failure_reason, failure_stage, created_at "
                "FROM dead_letter_queue WHERE resolved = FALSE ORDER BY created_at DESC LIMIT 20",
                conn,
            )
    except Exception:
        return pd.DataFrame()


def fetch_drift_snapshots(limit: int = 90) -> pd.DataFrame:
    try:
        with get_db_engine().connect() as conn:
            return pd.read_sql(
                "SELECT * FROM embedding_drift_snapshots ORDER BY created_at DESC LIMIT %(limit)s",
                conn,
                params={"limit": limit},
            )
    except Exception:
        return pd.DataFrame()


def fetch_retrieval_metrics(days: int = 7) -> pd.DataFrame:
    try:
        with get_db_engine().connect() as conn:
            return pd.read_sql(
                text(
                    "SELECT * FROM retrieval_metrics "
                    "WHERE created_at > NOW() - INTERVAL ':days days' "
                    "ORDER BY created_at DESC LIMIT 1000"
                ).bindparams(days=days),
                conn,
            )
    except Exception:
        return pd.DataFrame()


def fetch_documents(limit: int = 100) -> pd.DataFrame:
    try:
        with get_db_engine().connect() as conn:
            return pd.read_sql(
                "SELECT doc_id, source_uri, source_type, title, ingest_status, "
                "chunk_count, total_tokens, ingest_cost_usd, ingested_at "
                "FROM documents ORDER BY created_at DESC LIMIT %(limit)s",
                conn,
                params={"limit": limit},
            )
    except Exception:
        return pd.DataFrame()


def fetch_daily_cost_summary(days: int = 30) -> pd.DataFrame:
    return duckdb_query(
        f"SELECT * FROM daily_cost_summary ORDER BY summary_date DESC LIMIT {days}"
    )


def fetch_llm_interactions(limit: int = 200) -> pd.DataFrame:
    return duckdb_query(
        f"SELECT interaction_type, model, tokens_in, tokens_out, cost_usd, "
        f"latency_ms, status, created_at "
        f"FROM llm_interactions ORDER BY created_at DESC LIMIT {limit}"
    )


def fetch_top_expensive_requests(limit: int = 10) -> pd.DataFrame:
    return duckdb_query(
        f"SELECT request_id, interaction_type, model, tokens_in, tokens_out, "
        f"cost_usd, latency_ms, created_at "
        f"FROM llm_interactions ORDER BY cost_usd DESC LIMIT {limit}"
    )


def fetch_rca_interactions(limit: int = 5) -> pd.DataFrame:
    return duckdb_query(
        f"SELECT prompt_text, response_text, cost_usd, created_at "
        f"FROM llm_interactions WHERE interaction_type = 'rca_analysis' "
        f"ORDER BY created_at DESC LIMIT {limit}"
    )


def fetch_cache_stats() -> dict:
    try:
        r = get_redis()
        info = r.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 3) if total > 0 else 0.0,
        }
    except Exception:
        return {"hits": 0, "misses": 0, "hit_rate": 0.0}
