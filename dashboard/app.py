"""Streamlit dashboard entry point — LLM Data Platform."""
import streamlit as st

st.set_page_config(
    page_title="LLM Data Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.data_fetcher import fetch_api_health, fetch_drift_snapshots

health = fetch_api_health()
drift_df = fetch_drift_snapshots(limit=1)
latest_health = int(drift_df.iloc[0]["health_score"]) if not drift_df.empty and "health_score" in drift_df.columns else None

st.title("LLM Data Platform — Monitoring")

col1, col2, col3, col4 = st.columns(4)

with col1:
    api_ok = health.get("status") == "healthy"
    st.metric("API", "Online" if api_ok else "Degraded", delta=None)

with col2:
    db_ok = health.get("db") == "ok"
    st.metric("PostgreSQL", "Connected" if db_ok else "Error")

with col3:
    redis_ok = health.get("redis") == "ok"
    st.metric("Redis Cache", "Connected" if redis_ok else "Error")

with col4:
    if latest_health is not None:
        color = "normal" if latest_health >= 80 else ("off" if latest_health < 50 else "inverse")
        st.metric("Embedding Health", f"{latest_health}/100")
    else:
        st.metric("Embedding Health", "No data")

st.markdown("---")
st.markdown("""
### Navigation
Use the sidebar to explore:
- **Pipeline Health** — ingestion success rates, dead-letter queue
- **Token Costs** — daily spend, budget tracking
- **Retrieval Quality** — MMR diversity, cosine similarity
- **Embedding Drift** — centroid shift, health score, RCA
- **Ingestion Throughput** — chunks/tokens per run, cache hit rate
""")
