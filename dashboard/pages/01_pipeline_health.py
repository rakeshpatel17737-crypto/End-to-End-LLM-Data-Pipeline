import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Pipeline Health", layout="wide")
st.title("Pipeline Health")

from dashboard.data_fetcher import fetch_pipeline_runs, fetch_dlq_pending, fetch_api_health, fetch_drift_snapshots

health = fetch_api_health()
drift_df = fetch_drift_snapshots(limit=1)
runs_df = fetch_pipeline_runs(limit=30)
dlq_df = fetch_dlq_pending()

col1, col2, col3, col4 = st.columns(4)

with col1:
    score = int(drift_df.iloc[0]["health_score"]) if not drift_df.empty and "health_score" in drift_df.columns else None
    label = f"{score}/100" if score is not None else "No data"
    st.metric("Embedding Health Score", label)

with col2:
    st.metric("API Status", health.get("status", "unknown").upper())

with col3:
    total_runs = len(runs_df)
    st.metric("Pipeline Runs (30d)", total_runs)

with col4:
    pending_dlq = len(dlq_df)
    st.metric("DLQ Pending", pending_dlq, delta=None)

st.markdown("---")

if not runs_df.empty:
    st.subheader("Ingestion Runs — Success vs Failed")
    fig = px.bar(
        runs_df.sort_values("started_at"),
        x="started_at",
        y=["docs_succeeded", "docs_failed"],
        barmode="stack",
        color_discrete_map={"docs_succeeded": "#2ecc71", "docs_failed": "#e74c3c"},
        labels={"value": "Documents", "variable": "Outcome", "started_at": "Run Started"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pipeline runs yet. Trigger `make ingest` to start.")

st.markdown("---")
st.subheader(f"Dead-Letter Queue ({len(dlq_df)} pending)")

if not dlq_df.empty:
    st.dataframe(
        dlq_df[["source_uri", "source_type", "failure_stage", "failure_reason", "created_at"]].head(10),
        use_container_width=True,
    )
else:
    st.success("No failed documents in DLQ.")
