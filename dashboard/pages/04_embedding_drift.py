import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Embedding Drift", layout="wide")
st.title("Embedding Drift Monitor")

from dashboard.data_fetcher import fetch_drift_snapshots, fetch_rca_interactions

df = fetch_drift_snapshots(limit=90)
rca_df = fetch_rca_interactions(limit=3)

if not df.empty and "health_score" in df.columns:
    latest = df.iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = int(latest["health_score"])
        st.metric("Health Score", f"{score}/100")

    with col2:
        severity = latest.get("drift_severity", "ok")
        color_map = {"ok": "Normal", "warn": "Warning", "alert": "Critical"}
        st.metric("Drift Severity", severity.upper())

    with col3:
        centroid = latest.get("centroid_shift", 0)
        st.metric("Centroid Shift", f"{centroid:.4f}" if centroid is not None else "—")

    with col4:
        cosine = latest.get("cosine_sim_mean", 0)
        st.metric("Cosine Sim Mean", f"{cosine:.4f}" if cosine is not None else "—")

    severity_map = {"ok": "#2ecc71", "warn": "#f39c12", "alert": "#e74c3c"}
    df["severity_color"] = df["drift_severity"].map(severity_map).fillna("#95a5a6")

    st.markdown("---")
    st.subheader("Drift Signals Over Time")

    df_sorted = df.sort_values("created_at")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted["created_at"], y=df_sorted["centroid_shift"],
                             name="Centroid Shift", line=dict(color="#e74c3c")))
    fig.add_trace(go.Scatter(x=df_sorted["created_at"], y=df_sorted["cosine_sim_mean"],
                             name="Cosine Sim Mean", line=dict(color="#3498db")))
    fig.add_hline(y=0.05, line_dash="dash", line_color="#f39c12", annotation_text="warn threshold (shift)")
    fig.add_hline(y=0.15, line_dash="dash", line_color="#e74c3c", annotation_text="alert threshold (shift)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Health Score Timeline")
    fig2 = px.bar(
        df_sorted, x="created_at", y="health_score",
        color="drift_severity",
        color_discrete_map={"ok": "#2ecc71", "warn": "#f39c12", "alert": "#e74c3c"},
        labels={"health_score": "Health (0-100)", "created_at": "Date"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    if not rca_df.empty and "response_text" in rca_df.columns:
        st.markdown("---")
        st.subheader("Latest RCA Diagnosis")
        for _, row in rca_df.iterrows():
            with st.expander(f"RCA — {row.get('created_at', '')}"):
                st.markdown(row.get("response_text", "No details"))
else:
    st.info("No drift data yet. Run `make drift-check` to trigger the drift detection DAG.")
