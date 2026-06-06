import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Retrieval Quality", layout="wide")
st.title("RAG Retrieval Quality")

from dashboard.data_fetcher import fetch_retrieval_metrics

df = fetch_retrieval_metrics(days=7)

col1, col2, col3 = st.columns(3)

with col1:
    mean_div = df["mmr_diversity"].mean() if not df.empty and "mmr_diversity" in df.columns else None
    st.metric("Avg MMR Diversity", f"{mean_div:.3f}" if mean_div is not None else "No data")

with col2:
    mean_sim = df["mean_cosine_sim"].mean() if not df.empty and "mean_cosine_sim" in df.columns else None
    st.metric("Avg Cosine Similarity", f"{mean_sim:.3f}" if mean_sim is not None else "No data")

with col3:
    p95_lat = df["latency_ms"].quantile(0.95) if not df.empty and "latency_ms" in df.columns else None
    st.metric("p95 Latency (ms)", f"{p95_lat:.0f}" if p95_lat is not None else "No data")

st.markdown("---")

if not df.empty and "created_at" in df.columns:
    st.subheader("MMR Diversity & Cosine Similarity Over Time")
    fig = px.line(
        df.sort_values("created_at"),
        x="created_at",
        y=["mmr_diversity", "mean_cosine_sim"],
        labels={"value": "Score", "variable": "Metric", "created_at": "Time"},
        color_discrete_map={"mmr_diversity": "#9b59b6", "mean_cosine_sim": "#27ae60"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latency Distribution")
    fig2 = px.histogram(
        df, x="latency_ms", nbins=30,
        labels={"latency_ms": "Latency (ms)"},
        color_discrete_sequence=["#3498db"],
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Diversity vs Similarity")
    fig3 = px.scatter(
        df, x="mmr_diversity", y="mean_cosine_sim",
        color="latency_ms", color_continuous_scale="RdYlGn_r",
        labels={"mmr_diversity": "MMR Diversity", "mean_cosine_sim": "Cosine Similarity", "latency_ms": "Latency (ms)"},
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No retrieval metrics yet. Send a query to /query to populate this page.")
