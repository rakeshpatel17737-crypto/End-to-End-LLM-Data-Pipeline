import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Token Costs", layout="wide")
st.title("Token Costs & Budget")

from dashboard.data_fetcher import fetch_daily_cost_summary, fetch_top_expensive_requests, fetch_llm_interactions

cost_df = fetch_daily_cost_summary(days=30)
top_df = fetch_top_expensive_requests(limit=10)

col1, col2, col3 = st.columns(3)

today = cost_df.iloc[0] if not cost_df.empty else None

with col1:
    today_cost = float(today["total_cost_usd"]) if today is not None else 0.0
    st.metric("Today's Spend", f"${today_cost:.4f}")

with col2:
    budget = float(today["budget_limit_usd"]) if today is not None else 10.0
    st.metric("Daily Budget", f"${budget:.2f}")

with col3:
    exceeded = bool(today["budget_exceeded"]) if today is not None else False
    if exceeded:
        st.error("BUDGET EXCEEDED")
    else:
        remaining = budget - today_cost if today is not None else budget
        st.metric("Remaining", f"${remaining:.4f}")

if not cost_df.empty and "budget_exceeded" in cost_df.columns:
    exceeded_rows = cost_df[cost_df["budget_exceeded"] == True]
    if not exceeded_rows.empty:
        st.warning(f"Budget exceeded on {len(exceeded_rows)} day(s) in the last 30 days.")

st.markdown("---")

if not cost_df.empty:
    st.subheader("Daily Cost Breakdown")
    cost_df_sorted = cost_df.sort_values("summary_date")
    fig = go.Figure()
    for col, color, label in [
        ("embedding_cost", "#3498db", "Embeddings"),
        ("rag_cost", "#2ecc71", "RAG Queries"),
        ("rca_cost", "#e74c3c", "RCA Analysis"),
    ]:
        if col in cost_df_sorted.columns:
            fig.add_trace(go.Bar(
                x=cost_df_sorted["summary_date"],
                y=cost_df_sorted[col],
                name=label,
                marker_color=color,
            ))
    fig.update_layout(barmode="stack", xaxis_title="Date", yaxis_title="Cost (USD)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No cost data yet. Ingest documents and run a query to see costs.")

st.markdown("---")
st.subheader("Top 10 Most Expensive Requests")
if not top_df.empty:
    st.dataframe(top_df, use_container_width=True)
else:
    st.info("No LLM interactions logged yet.")
