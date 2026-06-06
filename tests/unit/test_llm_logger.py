"""Unit tests for warehouse/llm_logger.py"""
import os
import tempfile
import pytest


@pytest.fixture
def logger():
    db_path = tempfile.mktemp(suffix=".duckdb")
    from warehouse.llm_logger import LLMLogger
    yield LLMLogger(db_path)
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_log_and_retrieve(logger):
    iid = logger.log(
        request_id="req-001",
        interaction_type="embedding",
        model="all-MiniLM-L6-v2",
        tokens_in=100,
        tokens_out=0,
        cost_usd=0.000002,
        latency_ms=45.2,
        status="success",
    )
    assert iid is not None
    df = logger.query("SELECT * FROM llm_interactions")
    assert len(df) == 1
    assert df.iloc[0]["request_id"] == "req-001"
    assert df.iloc[0]["tokens_in"] == 100


def test_daily_summary_increments(logger):
    for _ in range(3):
        logger.log(
            request_id="req-x",
            interaction_type="embedding",
            model="all-MiniLM-L6-v2",
            tokens_in=50,
            tokens_out=0,
            cost_usd=0.000001,
            latency_ms=10.0,
        )
    df = logger.query("SELECT * FROM daily_cost_summary")
    assert len(df) == 1
    assert df.iloc[0]["total_calls"] == 3
    assert abs(df.iloc[0]["embedding_cost"] - 0.000003) < 1e-9


def test_budget_exceeded(logger, monkeypatch):
    monkeypatch.setenv("DAILY_COST_BUDGET_USD", "0.000001")
    logger.log(
        request_id="over",
        interaction_type="rag_query",
        model="llama-3.3-70b-versatile",
        tokens_in=1000,
        tokens_out=200,
        cost_usd=0.01,
        latency_ms=500.0,
    )
    assert logger.is_budget_exceeded()


def test_cached_status(logger):
    logger.log(
        request_id="r2",
        interaction_type="embedding",
        model="all-MiniLM-L6-v2",
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        latency_ms=0.5,
        status="cached",
    )
    df = logger.query("SELECT status FROM llm_interactions")
    assert df.iloc[0]["status"] == "cached"
