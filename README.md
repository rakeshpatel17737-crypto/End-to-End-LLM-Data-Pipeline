# End-to-End LLM Data Platform

A production-grade pipeline that ingests raw documents, generates embeddings, serves RAG queries, tracks embedding drift, and logs every LLM interaction with cost tracking — all observable through a live Streamlit dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                   │
│   PDFs / Web Pages / APIs                                         │
│         │                                                         │
│         ▼                                                         │
│   Airflow DAG ──► Chunker (512 tokens, overlap=50)               │
│                      │                                            │
│                      ▼                                            │
│               Quality Scorer ──► Dead-Letter Queue (on fail)     │
│                      │                                            │
│                      ▼                                            │
│            Redis Embedding Cache ──► OpenAI API (on miss)        │
│                      │                                            │
│                      ▼                                            │
│               pgvector (PostgreSQL 16)                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         RAG QUERY LAYER                          │
│                                                                   │
│   POST /query                                                     │
│       │                                                           │
│       ├──► Embed question (Redis cache → OpenAI)                 │
│       ├──► MMR Retrieval from pgvector (diversity + relevance)   │
│       ├──► Claude claude-sonnet-4-6 (prompt caching)             │
│       └──► Log to DuckDB (tokens, cost, latency, request_id)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DRIFT DETECTION (Daily)                      │
│                                                                   │
│   Airflow DAG                                                     │
│       ├──► Sample 500 current embeddings from pgvector           │
│       ├──► Load 500 baseline embeddings (age > 30 days)          │
│       ├──► Centroid shift + KS test + Z-score on norms           │
│       ├──► Health score 0–100                                     │
│       └──► Claude RCA (if drifting) → log diagnosis              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                         │
│                                                                   │
│   01 Pipeline Health    ── run success/fail, DLQ pending         │
│   02 Token Costs        ── daily spend, budget alerts            │
│   03 Retrieval Quality  ── MMR diversity, cosine similarity      │
│   04 Embedding Drift    ── centroid shift, health score, RCA     │
│   05 Ingestion Throughput── chunks/tokens per run, cache hit rate│
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow 2.9.2 |
| Ingestion | PyMuPDF (PDF), httpx + BeautifulSoup (web), httpx (API) |
| Chunking | tiktoken cl100k\_base, 512 tokens, 50-token overlap |
| Chunk Quality | langdetect, regex PII detection, MinHash dedup |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Embedding Cache | Redis — SHA-256 keyed, 7-day TTL |
| Vector Store | PostgreSQL 16 + pgvector (IVFFlat cosine index) |
| RAG Retrieval | Greedy MMR (Maximal Marginal Relevance) |
| LLM | Anthropic Claude `claude-sonnet-4-6` with prompt caching |
| LLM Warehouse | DuckDB — all interactions, costs, latency |
| Drift Detection | Centroid shift, KS test, Z-score on embedding norms |
| RCA | Claude tool use + rule-based fallback (tenacity retry) |
| Dashboard | Streamlit + Plotly |
| Infrastructure | Docker Compose (7 services) |

---

## Services

| Service | Port | Description |
|---|---|---|
| `postgres` (pgvector) | 5432 | Chunks, embeddings, drift snapshots, pipeline runs |
| `redis` | 6379 | Embedding cache + pipeline state |
| `airflow-webserver` | 8080 | DAG UI (admin / admin) |
| `airflow-scheduler` | — | DAG execution |
| `api` | 8000 | FastAPI RAG endpoint |
| `dashboard` | 8501 | Streamlit 5-page monitoring |

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/rakeshpatel17737-crypto/End-to-End-LLM-Data-Pipeline.git
cd End-to-End-LLM-Data-Pipeline
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
AIRFLOW_FERNET_KEY=   # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW_SECRET_KEY=   # any random string
```

### 2. Start all services

```bash
make up
```

| URL | Service |
|---|---|
| http://localhost:8501 | Streamlit Dashboard |
| http://localhost:8000/docs | FastAPI Swagger UI |
| http://localhost:8080 | Airflow (admin / admin) |

### 3. Ingest documents

```bash
make ingest
```

This triggers the Airflow DAG which fetches 3 Wikipedia articles (Apache Airflow, RAG, LLMs), chunks them, embeds via OpenAI, and stores in pgvector.

### 4. Run a RAG query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Retrieval-Augmented Generation?", "top_k": 3}'
```

### 5. Check embedding drift

```bash
make drift-check
```

### 6. View cost report

```bash
make cost-report
```

---

## Project Structure

```
llm-data-platform/
├── airflow/dags/
│   ├── ingest_documents_dag.py   # PDF / web / API ingestion pipeline
│   └── embedding_drift_dag.py    # Daily drift detection + RCA + budget check
├── ingestion/
│   ├── loaders/                  # pdf_loader, web_loader, api_loader
│   ├── chunker.py                # tiktoken chunking + SHA-256 + MinHash
│   └── quality_scorer.py         # PII detection, langdetect, dedup
├── embeddings/
│   ├── embedder.py               # OpenAI batch embed + Redis cache + cost log
│   ├── vector_store.py           # pgvector upsert + MMR retrieval
│   └── embedding_cache.py        # Redis SHA-256 cache
├── drift/
│   ├── statistical_tests.py      # KS test, Z-score, KL divergence
│   ├── embedding_drift_detector.py # Centroid shift, health score 0-100
│   └── rca_engine.py             # Claude RCA with tool use + fallback
├── warehouse/
│   ├── llm_logger.py             # DuckDB writer: all LLM interactions + costs
│   └── schemas.py                # DuckDB DDL
├── api/
│   ├── main.py                   # FastAPI + request_id middleware
│   ├── rag_engine.py             # Full RAG pipeline
│   └── schemas.py                # Pydantic request/response models
├── dashboard/
│   ├── app.py                    # Streamlit entry point
│   ├── data_fetcher.py           # PostgreSQL + DuckDB + Redis data layer
│   └── pages/                    # 5 monitoring pages
├── docker/
│   ├── init-postgres.sql         # pgvector schema
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── tests/unit/                   # 19 unit tests (all passing)
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## Key Features

### Smart Chunking
Token-aware sliding window (512 tokens, 50-token overlap) using tiktoken. Each chunk gets SHA-256 fingerprinting for exact dedup and MinHash signatures for near-duplicate detection.

### Embedding Cache
Redis cache keyed on SHA-256(chunk text). Identical content across different documents hits the cache — avoids re-calling the OpenAI API and cuts costs significantly on repeated ingestion.

### MMR Retrieval
Greedy Maximal Marginal Relevance retrieval balances relevance and diversity. Fetches top-50 by cosine similarity, then iteratively selects chunks that maximize `λ·sim(chunk, query) - (1-λ)·max(sim(chunk, selected))`.

### DuckDB LLM Warehouse
Every embedding call, RAG query, and RCA analysis is logged to DuckDB with: `request_id`, model, tokens\_in, tokens\_out, cost\_usd, latency\_ms. Daily cost summaries with budget\_exceeded flag. Shared volume between API and dashboard containers.

### Embedding Drift Detection
Three signals: centroid shift (cosine distance between current and baseline mean), KS test on L2 norm distributions, and Z-score on norm means. Health score 0–100 (adapted from production ML observability patterns).

### LLM-Powered RCA
When drift is detected, Claude (`claude-sonnet-4-6`) diagnoses the root cause using structured tool use with prompt caching on the system prompt. Falls back to rule-based diagnosis if the API is unavailable. Cause categories: `topic_distribution_shift`, `data_quality_degradation`, `seasonal_content_shift`, and more.

---

## Makefile Reference

```bash
make up              # Start all 7 services
make down            # Stop all services
make down-volumes    # Stop and delete all data
make logs            # Tail all service logs
make logs-api        # Tail a specific service
make ingest          # Trigger ingestion DAG
make drift-check     # Trigger drift detection DAG
make cost-report     # Print DuckDB daily cost summary
make cache-stats     # Redis cache hit/miss stats
make test            # Run all tests
make test-unit       # Run unit tests only (no Docker needed)
make shell-api       # Shell into API container
make shell-postgres  # psql into PostgreSQL
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | RAG query — returns answer + source chunks + metrics |
| `GET` | `/health` | Service health check |
| `GET` | `/metrics/cost` | Last 30 days cost from DuckDB |
| `GET` | `/metrics/drift` | Last 30 drift snapshots |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for embeddings |
| `ANTHROPIC_API_KEY` | Anthropic API key for RAG + RCA |
| `POSTGRES_USER/PASSWORD/DB` | PostgreSQL credentials |
| `AIRFLOW_FERNET_KEY` | Airflow encryption key |
| `AIRFLOW_SECRET_KEY` | Airflow webserver secret |
| `DAILY_COST_BUDGET_USD` | Budget alert threshold (default: $10) |

---

## Running Tests

No Docker required for unit tests:

```bash
pip install tiktoken duckdb numpy langdetect datasketch pytest
python -m pytest tests/unit/ -v
```

19 tests covering: chunker, quality scorer, drift detector, and LLM logger.
