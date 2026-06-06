import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ingestion Throughput", layout="wide")
st.title("Ingestion Throughput")

from dashboard.data_fetcher import fetch_pipeline_runs, fetch_documents, fetch_cache_stats

runs_df = fetch_pipeline_runs(limit=30)
docs_df = fetch_documents(limit=100)
cache_stats = fetch_cache_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_chunks = int(runs_df["chunks_created"].sum()) if not runs_df.empty else 0
    st.metric("Total Chunks Created", f"{total_chunks:,}")

with col2:
    total_tokens = int(runs_df["tokens_embedded"].sum()) if not runs_df.empty else 0
    st.metric("Total Tokens Embedded", f"{total_tokens:,}")

with col3:
    st.metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.1%}")

with col4:
    total_docs = len(docs_df)
    st.metric("Total Documents", total_docs)

st.markdown("---")

if not runs_df.empty:
    runs_sorted = runs_df.sort_values("started_at")

    st.subheader("Chunks & Tokens per Run")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=runs_sorted["started_at"], y=runs_sorted["chunks_created"],
        name="Chunks Created", marker_color="#3498db",
    ))
    fig.add_trace(go.Scatter(
        x=runs_sorted["started_at"], y=runs_sorted["tokens_embedded"],
        name="Tokens Embedded", yaxis="y2", line=dict(color="#e74c3c"),
    ))
    fig.update_layout(
        yaxis=dict(title="Chunks"),
        yaxis2=dict(title="Tokens", overlaying="y", side="right"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Embedding Cost per Run")
    fig2 = px.line(
        runs_sorted, x="started_at", y="embed_cost_usd",
        labels={"embed_cost_usd": "Cost (USD)", "started_at": "Run"},
        markers=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("Document Status")

if not docs_df.empty:
    status_counts = docs_df["ingest_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig3 = px.pie(status_counts, values="Count", names="Status",
                  color_discrete_map={"done": "#2ecc71", "failed": "#e74c3c",
                                      "pending": "#f39c12", "processing": "#3498db"})
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.plotly_chart(fig3)
    with col_b:
        st.dataframe(
            docs_df[["title", "source_type", "ingest_status", "chunk_count", "ingest_cost_usd", "ingested_at"]],
            use_container_width=True,
        )
else:
    st.info("No documents ingested yet.")

st.markdown("---")
col_x, col_y = st.columns(2)
with col_x:
    st.metric("Redis Cache Hits", f"{cache_stats['hits']:,}")
with col_y:
    st.metric("Redis Cache Misses", f"{cache_stats['misses']:,}")
