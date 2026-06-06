"""Airflow DAG: daily embedding drift detection + LLM RCA + cost budget check.

Task graph:
  sample_current_embeddings
    → load_baseline_embeddings
        → compute_drift_metrics
            → branch_on_severity [Branch]
                → log_healthy_snapshot
                → llm_rca_analysis → log_drift_snapshot
            → cost_budget_check
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

sys.path.insert(0, "/opt/llm_platform")

DEFAULT_ARGS = {
    "owner": "llm-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


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


def sample_current_embeddings(ti, **kwargs):
    from embeddings.vector_store import fetch_embedding_sample
    conn = _pg_conn()
    embeddings = fetch_embedding_sample(conn, n=500, age_days_max=7)
    conn.close()
    ti.xcom_push(key="current_embeddings", value=embeddings)
    print(f"Sampled {len(embeddings)} current embeddings")


def load_baseline_embeddings(ti, **kwargs):
    from embeddings.vector_store import fetch_baseline_embeddings
    conn = _pg_conn()
    embeddings = fetch_baseline_embeddings(conn, n=500, age_days_min=7)
    conn.close()

    if not embeddings:
        current = ti.xcom_pull(task_ids="sample_current_embeddings", key="current_embeddings") or []
        embeddings = current[:100] if len(current) > 100 else current
        print(f"No baseline found, using {len(embeddings)} current embeddings as baseline")

    ti.xcom_push(key="baseline_embeddings", value=embeddings)
    print(f"Loaded {len(embeddings)} baseline embeddings")


def compute_drift_metrics(ti, **kwargs):
    from drift.embedding_drift_detector import detect_embedding_drift

    current = ti.xcom_pull(task_ids="sample_current_embeddings", key="current_embeddings") or []
    baseline = ti.xcom_pull(task_ids="load_baseline_embeddings", key="baseline_embeddings") or []

    result = detect_embedding_drift(current, baseline)
    result_dict = {
        "centroid_shift": result.centroid_shift,
        "norm_iqr": result.norm_iqr,
        "norm_mean": result.norm_mean,
        "norm_std": result.norm_std,
        "cosine_sim_mean": result.cosine_sim_mean,
        "cosine_sim_std": result.cosine_sim_std,
        "ks_statistic": result.ks_statistic,
        "ks_p_value": result.ks_p_value,
        "z_score_norm_mean": result.z_score_norm_mean,
        "drift_severity": result.drift_severity,
        "health_score": result.health_score,
        "chunk_sample_size": result.chunk_sample_size,
        "baseline_size": result.baseline_size,
    }
    ti.xcom_push(key="drift_result", value=result_dict)
    print(f"Drift: severity={result.drift_severity}, health={result.health_score}, centroid_shift={result.centroid_shift:.4f}")


def branch_on_severity(ti, **kwargs):
    result = ti.xcom_pull(task_ids="compute_drift_metrics", key="drift_result") or {}
    severity = result.get("drift_severity", "ok")
    if severity in ("warn", "alert"):
        return "llm_rca_analysis"
    return "log_healthy_snapshot"


def llm_rca_analysis(ti, **kwargs):
    from drift.embedding_drift_detector import EmbeddingDriftResult
    from drift.rca_engine import rca_engine

    result_dict = ti.xcom_pull(task_ids="compute_drift_metrics", key="drift_result") or {}
    result = EmbeddingDriftResult(**result_dict)

    diagnosis = asyncio.run(rca_engine.analyze(result))

    ti.xcom_push(key="rca_diagnosis", value={
        "probable_cause": diagnosis.probable_cause,
        "cause_category": diagnosis.cause_category,
        "confidence": diagnosis.confidence,
        "remediation_steps": diagnosis.remediation_steps,
        "urgency": diagnosis.urgency,
        "estimated_impact": diagnosis.estimated_impact,
        "model_used": diagnosis.model_used,
        "tokens_used": diagnosis.tokens_used,
        "fallback_used": diagnosis.fallback_used,
    })

    print(f"RCA: {diagnosis.cause_category} ({diagnosis.urgency}) — {diagnosis.probable_cause[:100]}")


def log_drift_snapshot(ti, **kwargs):
    result_dict = ti.xcom_pull(task_ids="compute_drift_metrics", key="drift_result") or {}
    rca = ti.xcom_pull(task_ids="llm_rca_analysis", key="rca_diagnosis") or {}
    rca_summary = rca.get("probable_cause", "")

    _write_snapshot(result_dict, rca_summary)


def log_healthy_snapshot(ti, **kwargs):
    result_dict = ti.xcom_pull(task_ids="compute_drift_metrics", key="drift_result") or {}
    _write_snapshot(result_dict, rca_summary="")


def _write_snapshot(result_dict: dict, rca_summary: str) -> None:
    conn = _pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embedding_drift_snapshots
                    (snapshot_date, centroid_shift, norm_iqr, norm_mean, norm_std,
                     cosine_sim_mean, cosine_sim_std, ks_statistic, ks_p_value,
                     z_score_norm_mean, drift_severity, health_score,
                     chunk_sample_size, baseline_size, rca_summary)
                VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    result_dict.get("centroid_shift"),
                    result_dict.get("norm_iqr"),
                    result_dict.get("norm_mean"),
                    result_dict.get("norm_std"),
                    result_dict.get("cosine_sim_mean"),
                    result_dict.get("cosine_sim_std"),
                    result_dict.get("ks_statistic"),
                    result_dict.get("ks_p_value"),
                    result_dict.get("z_score_norm_mean"),
                    result_dict.get("drift_severity", "ok"),
                    result_dict.get("health_score", 100),
                    result_dict.get("chunk_sample_size", 0),
                    result_dict.get("baseline_size", 0),
                    rca_summary[:500] if rca_summary else None,
                ],
            )
        conn.commit()
        print(f"Drift snapshot written: severity={result_dict.get('drift_severity')}, health={result_dict.get('health_score')}")
    except Exception as exc:
        conn.rollback()
        print(f"Failed to write drift snapshot: {exc}")
    finally:
        conn.close()


def cost_budget_check(ti, **kwargs):
    from warehouse.llm_logger import LLMLogger
    duckdb_path = os.environ.get("DUCKDB_PATH", "/data/warehouse/llm_logs.duckdb")
    llm_logger = LLMLogger(duckdb_path)

    if llm_logger.is_budget_exceeded():
        df = llm_logger.query(
            "SELECT total_cost_usd, budget_limit_usd FROM daily_cost_summary WHERE summary_date = current_date"
        )
        if not df.empty:
            row = df.iloc[0]
            print(f"⚠️  BUDGET EXCEEDED: ${row['total_cost_usd']:.4f} / ${row['budget_limit_usd']:.2f} daily limit")
    else:
        df = llm_logger.query(
            "SELECT COALESCE(total_cost_usd, 0) as cost FROM daily_cost_summary WHERE summary_date = current_date"
        )
        cost = df.iloc[0]["cost"] if not df.empty else 0.0
        print(f"Budget OK: ${cost:.4f} spent today")


with DAG(
    dag_id="embedding_drift_dag",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["llm-platform", "drift"],
) as dag:

    t_current = PythonOperator(task_id="sample_current_embeddings", python_callable=sample_current_embeddings)
    t_baseline = PythonOperator(task_id="load_baseline_embeddings", python_callable=load_baseline_embeddings)
    t_compute = PythonOperator(task_id="compute_drift_metrics", python_callable=compute_drift_metrics)
    t_branch = BranchPythonOperator(task_id="branch_on_severity", python_callable=branch_on_severity)
    t_healthy = PythonOperator(task_id="log_healthy_snapshot", python_callable=log_healthy_snapshot)
    t_rca = PythonOperator(task_id="llm_rca_analysis", python_callable=llm_rca_analysis)
    t_snapshot = PythonOperator(task_id="log_drift_snapshot", python_callable=log_drift_snapshot)
    t_budget = PythonOperator(task_id="cost_budget_check", python_callable=cost_budget_check)

    t_current >> t_baseline >> t_compute >> t_branch >> [t_healthy, t_rca]
    t_rca >> t_snapshot
    [t_healthy, t_snapshot] >> t_budget
