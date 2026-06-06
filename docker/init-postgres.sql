CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uri      TEXT NOT NULL UNIQUE,
    source_type     TEXT NOT NULL,
    title           TEXT,
    language        TEXT,
    ingest_status   TEXT NOT NULL DEFAULT 'pending',
    chunk_count     INT DEFAULT 0,
    total_tokens    INT DEFAULT 0,
    ingest_cost_usd NUMERIC(10, 6) DEFAULT 0,
    ingested_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    embedding       vector(384),
    token_count     INT NOT NULL,
    source_uri      TEXT NOT NULL,
    page_number     INT,
    language        TEXT,
    quality_score   NUMERIC(5, 4),
    has_pii         BOOLEAN DEFAULT FALSE,
    minhash_sig     TEXT,
    embedded_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS dead_letter_queue (
    dlq_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uri      TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    failure_reason  TEXT NOT NULL,
    failure_stage   TEXT NOT NULL,
    attempt_count   INT DEFAULT 1,
    last_attempt_at TIMESTAMPTZ DEFAULT NOW(),
    next_retry_at   TIMESTAMPTZ,
    raw_payload     JSONB,
    resolved        BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embedding_drift_snapshots (
    snapshot_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date       DATE NOT NULL,
    centroid_shift      NUMERIC(10, 6),
    norm_iqr            NUMERIC(10, 6),
    norm_mean           NUMERIC(10, 6),
    norm_std            NUMERIC(10, 6),
    cosine_sim_mean     NUMERIC(10, 6),
    cosine_sim_std      NUMERIC(10, 6),
    ks_statistic        NUMERIC(10, 6),
    ks_p_value          NUMERIC(10, 6),
    z_score_norm_mean   NUMERIC(10, 6),
    drift_severity      TEXT,
    health_score        INT,
    chunk_sample_size   INT,
    baseline_size       INT,
    rca_summary         TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dag_id          TEXT NOT NULL,
    airflow_run_id  TEXT UNIQUE,
    status          TEXT NOT NULL,
    docs_attempted  INT DEFAULT 0,
    docs_succeeded  INT DEFAULT 0,
    docs_failed     INT DEFAULT 0,
    chunks_created  INT DEFAULT 0,
    tokens_embedded INT DEFAULT 0,
    embed_cost_usd  NUMERIC(10, 6) DEFAULT 0,
    duration_sec    NUMERIC(10, 2),
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS retrieval_metrics (
    metric_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id      TEXT NOT NULL,
    query_text      TEXT NOT NULL,
    top_k           INT,
    mmr_diversity   NUMERIC(6, 4),
    mean_cosine_sim NUMERIC(6, 4),
    result_count    INT,
    latency_ms      NUMERIC(10, 2),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents(ingest_status);
CREATE INDEX IF NOT EXISTS drift_snapshots_date_idx ON embedding_drift_snapshots(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS pipeline_runs_dag_idx ON pipeline_runs(dag_id, started_at DESC);
