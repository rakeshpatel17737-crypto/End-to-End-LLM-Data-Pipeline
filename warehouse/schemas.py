LLM_LOGS_DDL = """
CREATE TABLE IF NOT EXISTS llm_interactions (
    interaction_id  VARCHAR PRIMARY KEY,
    request_id      VARCHAR NOT NULL,
    interaction_type VARCHAR NOT NULL,
    model           VARCHAR NOT NULL,
    prompt_text     VARCHAR,
    response_text   VARCHAR,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    cost_usd        DOUBLE NOT NULL DEFAULT 0.0,
    latency_ms      DOUBLE NOT NULL,
    status          VARCHAR NOT NULL DEFAULT 'success',
    error_message   VARCHAR,
    metadata        VARCHAR,
    created_at      TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS daily_cost_summary (
    summary_date    DATE PRIMARY KEY,
    total_cost_usd  DOUBLE NOT NULL DEFAULT 0.0,
    embedding_cost  DOUBLE NOT NULL DEFAULT 0.0,
    rag_cost        DOUBLE NOT NULL DEFAULT 0.0,
    rca_cost        DOUBLE NOT NULL DEFAULT 0.0,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    total_calls     INTEGER NOT NULL DEFAULT 0,
    budget_limit_usd DOUBLE DEFAULT 10.0,
    budget_exceeded BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMP DEFAULT current_timestamp
);
"""
