"""FastAPI RAG endpoint with lifespan resource management and request_id middleware.

Patterns reused from:
- llm-sql-codegen/api/main.py: lifespan context manager
- feature-store/api/middleware.py: request_id injection
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import anthropic
import psycopg2
import redis as redis_lib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg2 import register_vector

from api.config import config
from api.schemas import QueryRequest, QueryResponse, HealthResponse, SourceChunk
from api.rag_engine import run_rag_query
from embeddings.embedding_cache import EmbeddingCache
from warehouse.llm_logger import LLMLogger
from shared.logging import configure_logging, get_logger

configure_logging(config.log_level)
logger = get_logger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LLM Data Platform API starting", environment=config.environment)

    conn = psycopg2.connect(config.postgres_dsn)
    register_vector(conn)
    _state["pg_conn"] = conn

    _state["redis"] = redis_lib.Redis(
        host=config.redis_host, port=config.redis_port, decode_responses=False
    )
    _state["cache"] = EmbeddingCache(_state["redis"])
    _state["llm_logger"] = LLMLogger(config.duckdb_path)
    _state["anthropic"] = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    logger.info("All connections initialized")
    yield

    _state["pg_conn"].close()
    _state["redis"].close()
    logger.info("Connections closed")


app = FastAPI(
    title="LLM Data Platform RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id
    import time
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = str(latency_ms)
    return response


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    try:
        result = await run_rag_query(
            question=req.question,
            pg_conn=_state["pg_conn"],
            cache=_state["cache"],
            llm_logger=_state["llm_logger"],
            anthropic_client=_state["anthropic"],
            anthropic_model=config.anthropic_model,
            top_k=req.top_k,
            mmr_lambda=req.mmr_lambda,
            request_id=request_id,
        )
    except Exception as exc:
        logger.error("RAG query failed", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result.answer,
        source_chunks=[
            SourceChunk(
                chunk_id=str(c.get("chunk_id", "")),
                content=c.get("content", ""),
                source_uri=c.get("source_uri", ""),
                page_number=c.get("page_number"),
                cosine_sim=c.get("cosine_sim"),
            )
            for c in result.source_chunks
        ],
        mmr_diversity=result.mmr_diversity,
        mean_cosine_sim=result.mean_cosine_sim,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        request_id=result.request_id,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    db_status = "ok"
    try:
        with _state["pg_conn"].cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        db_status = "error"

    redis_status = "ok"
    try:
        _state["redis"].ping()
    except Exception:
        redis_status = "error"

    duckdb_status = "ok"
    try:
        _state["llm_logger"].query("SELECT 1")
    except Exception:
        duckdb_status = "error"

    overall = "healthy" if all(s == "ok" for s in [db_status, redis_status, duckdb_status]) else "degraded"
    return HealthResponse(status=overall, db=db_status, redis=redis_status, duckdb=duckdb_status)


@app.get("/metrics/cost")
def cost_metrics():
    df = _state["llm_logger"].query(
        "SELECT * FROM daily_cost_summary ORDER BY summary_date DESC LIMIT 30"
    )
    return df.to_dict(orient="records")


@app.get("/metrics/drift")
def drift_metrics(conn=None):
    try:
        import pandas as pd
        from sqlalchemy import create_engine, text
        engine = create_engine(config.postgres_dsn)
        with engine.connect() as c:
            df = pd.read_sql(
                "SELECT * FROM embedding_drift_snapshots ORDER BY created_at DESC LIMIT 30",
                c,
            )
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
