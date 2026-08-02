"""
Streamlit investigator dashboard for the AML Smurfing Detection Engine.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

st.set_page_config(page_title="AML Smurfing Detection Engine", layout="wide", page_icon="🛡️")

st.title("🛡️ AML Smurfing Detection Engine")
st.caption("Graph-based unsupervised anomaly detection for structured money-laundering rings")

df = pd.read_csv("data/account_risk_scores_final.csv")
G = nx.read_gexf("data/transaction_graph.gexf")

with open("outputs/metrics_after_propagation.json") as f:
    metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accounts Screened", f"{len(df):,}")
col2.metric("Flagged (Seed + Propagation)", metrics["total_accounts_flagged_after_propagation"])
col3.metric("Recall vs Ground Truth", f"{metrics['recall_after_propagation']:.0%}")
col4.metric("Precision", f"{metrics['precision_after_propagation']:.0%}")

st.divider()

left, right = st.columns([1, 2])

with left:
    st.subheader("🔍 Screen an Account")
    account_id = st.text_input("Account ID", value="SINK_00")
    if account_id in df["account_id"].values:
        r = df[df["account_id"] == account_id].iloc[0]
        st.metric("Risk Score", f"{r['risk_score']:.3f}")
        st.metric("Risk Percentile", f"{r['risk_percentile']:.1f}")
        badge = {"CRITICAL": "🔴", "WARNING": "🟡", "NORMAL": "🟢"}.get(r["alert_status"], "⚪")
        st.write(f"**Status:** {badge} {r['alert_status']}")
        st.write(f"**Ground-truth role (synthetic label):** {r['role']}")
    else:
        st.warning("Account not found. Try one from the Top Alerts table.")

    st.subheader("🚨 Top Alerts")
    top = df.sort_values("risk_score", ascending=False).head(15)
    st.dataframe(
        top[["account_id", "role", "risk_score", "alert_status"]],
        use_container_width=True, hide_index=True,
    )

with right:
    st.subheader("🕸️ Network Snapshot")
    selected_ring = st.selectbox("View ring", [f"{i:02d}" for i in range(8)])
    ring_nodes = set(df[df["account_id"].str.contains(f"KING_{selected_ring}|MULE_{selected_ring}_|SINK_{selected_ring}", regex=True, na=False)]["account_id"])
    context_nodes = set()
    for n in ring_nodes:
        if n in G:
            context_nodes.update(G.predecessors(n))
            context_nodes.update(G.successors(n))
    subG = G.subgraph(ring_nodes | context_nodes).copy()

    colors = {"normal": "#4C9AFF", "mule": "#FFAB00", "kingpin": "#DE350B", "sink": "#6554C0"}
    role_map = dict(zip(df["account_id"], df["role"]))
    node_colors = [colors.get(role_map.get(n, "normal"), "#4C9AFF") for n in subG.nodes()]
    node_sizes = [500 if role_map.get(n, "normal") in ("kingpin", "sink") else 80 for n in subG.nodes()]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(subG, seed=42, k=0.4)
    nx.draw_networkx_edges(subG, pos, alpha=0.3, arrows=True, arrowsize=6, ax=ax)
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    patches = [mpatches.Patch(color=c, label=r) for r, c in colors.items()]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title(f"Ring #{selected_ring}: Kingpin → Mules → Offshore Sink")
    ax.axis("off")
    st.pyplot(fig)

st.divider()
st.subheader("📊 Risk Score Distribution by Role")
st.image("outputs/figures/risk_score_distribution.png", use_container_width=True)
