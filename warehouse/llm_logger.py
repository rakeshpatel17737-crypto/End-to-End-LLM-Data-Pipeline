"""DuckDB-backed logger for all LLM interactions — embeddings, RAG queries, RCA analyses."""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

import duckdb
import pandas as pd

from .schemas import LLM_LOGS_DDL

_COST_COLUMN = {
    "embedding": "embedding_cost",
    "rag_query": "rag_cost",
    "rca_analysis": "rca_cost",
}


class LLMLogger:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with duckdb.connect(self._db_path) as con:
            con.execute(LLM_LOGS_DDL)

    def log(
        self,
        *,
        request_id: str,
        interaction_type: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float,
        status: str = "success",
        prompt_text: str | None = None,
        response_text: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        interaction_id = str(uuid.uuid4())
        meta_str = json.dumps(metadata) if metadata else None

        with duckdb.connect(self._db_path) as con:
            con.execute(
                """
                INSERT INTO llm_interactions
                (interaction_id, request_id, interaction_type, model, prompt_text,
                 response_text, tokens_in, tokens_out, cost_usd, latency_ms,
                 status, error_message, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
                """,
                [
                    interaction_id, request_id, interaction_type, model,
                    prompt_text, response_text, tokens_in, tokens_out,
                    cost_usd, latency_ms, status, error_message, meta_str,
                ],
            )
            self._upsert_daily_summary(con, cost_usd, tokens_in + tokens_out, interaction_type)

        return interaction_id

    def _upsert_daily_summary(
        self, con: duckdb.DuckDBPyConnection, cost_usd: float, tokens: int, itype: str
    ) -> None:
        cost_col = _COST_COLUMN.get(itype, "rag_cost")
        budget = float(os.getenv("DAILY_COST_BUDGET_USD", "10.0"))

        exceeded = cost_usd > budget
        con.execute(
            f"""
            INSERT INTO daily_cost_summary
                (summary_date, total_cost_usd, {cost_col}, total_tokens, total_calls,
                 budget_limit_usd, budget_exceeded)
            VALUES (current_date, ?, ?, ?, 1, ?, ?)
            ON CONFLICT (summary_date) DO UPDATE SET
                total_cost_usd  = daily_cost_summary.total_cost_usd + excluded.total_cost_usd,
                {cost_col}      = daily_cost_summary.{cost_col} + excluded.{cost_col},
                total_tokens    = daily_cost_summary.total_tokens + excluded.total_tokens,
                total_calls     = daily_cost_summary.total_calls + 1,
                budget_exceeded = (daily_cost_summary.total_cost_usd + excluded.total_cost_usd)
                                   > daily_cost_summary.budget_limit_usd,
                updated_at      = now()
            """,
            [cost_usd, cost_usd, tokens, budget, exceeded],
        )

    def query(self, sql: str) -> pd.DataFrame:
        with duckdb.connect(self._db_path, read_only=True) as con:
            return con.execute(sql).df()

    def is_budget_exceeded(self) -> bool:
        df = self.query(
            "SELECT budget_exceeded FROM daily_cost_summary WHERE summary_date = current_date"
        )
        if df.empty:
            return False
        return bool(df.iloc[0]["budget_exceeded"])
