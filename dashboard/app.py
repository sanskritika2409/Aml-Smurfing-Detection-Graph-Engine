"""
Streamlit dashboard for the Customer Digital Twin Engine.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json

st.set_page_config(page_title="Customer Digital Twin", layout="wide", page_icon="👤")

st.title("👤 Customer Digital Twin")
st.caption("Unified churn + lifetime value + recommendation engine, one API call per customer")

df = pd.read_csv("data/digital_twin.csv")
bundle = joblib.load("outputs/digital_twin_model.joblib")
churn_model, ltv_model, encoders, feature_cols = (
    bundle["churn_model"], bundle["ltv_model"], bundle["encoders"], bundle["feature_cols"]
)

with open("outputs/metrics.json") as f:
    metrics = json.load(f)
with open("outputs/recommender_metrics.json") as f:
    rec_metrics = json.load(f)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers Scored", f"{len(df):,}")
col2.metric("Churn Model AUC", metrics["churn_model"]["auc_roc"])
col3.metric("LTV MAPE", f"{metrics['ltv_model']['mape_on_active_customers']:.1%}")
col4.metric("Recommender Lift", f"{rec_metrics['lift_over_random']}x vs random")

st.divider()
left, right = st.columns([1, 1.3])

with left:
    st.subheader("🔍 Look Up a Customer")
    user_id = st.selectbox("Customer ID", df["user_id"].tolist(), index=0)
    r = df[df["user_id"] == user_id].iloc[0]

    action_colors = {
        "RETENTION_OFFER": "🔴", "UPSELL": "🟢", "LOW_COST_NURTURE": "🟡", "STANDARD_NEWSLETTER": "🔵",
    }
    st.markdown(f"### {action_colors.get(r['system_action'], '⚪')} {r['system_action'].replace('_', ' ').title()}")
    st.write(r["action_message"])

    m1, m2 = st.columns(2)
    m1.metric("Churn Risk", f"{r['churn_score']:.0%}")
    m2.metric("Predicted 12m LTV", f"${r['predicted_ltv_12m']:,.0f}")
    st.write(f"**Recommended categories:** {r['recommended_categories']}")
    st.write(f"**Plan:** {r['plan_type']} | **Region:** {r['region']}")

    st.subheader("🧪 What-If Simulator")
    st.caption("Adjust engagement variables to see the churn score respond in real time")
    sim_frequency = st.slider("Sessions in last 30 days", 0, 20, int(r["frequency_30d"]))
    sim_recency = st.slider("Days since last visit", 0, 120, int(min(r["recency_days"], 120)))
    sim_duration = st.slider("Avg session duration (min)", 0.0, 20.0, float(min(r["avg_session_duration"], 20)))

    sim_row = r[feature_cols].copy()
    sim_row["frequency_30d"] = sim_frequency
    sim_row["recency_days"] = sim_recency
    sim_row["avg_session_duration"] = sim_duration
    sim_row["recent_avg_duration"] = sim_duration

    sim_X = pd.DataFrame([sim_row])[feature_cols]
    new_churn = churn_model.predict_proba(sim_X)[:, 1][0]

    delta = new_churn - r["churn_score"]
    st.metric("Simulated Churn Score", f"{new_churn:.0%}", delta=f"{delta:+.0%}", delta_color="inverse")

with right:
    st.subheader("📊 Churn Model Performance")
    st.image("outputs/figures/churn_roc_curve.png", use_container_width=True)
    st.image("outputs/figures/churn_score_distribution.png", use_container_width=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("💰 LTV Prediction Quality")
    st.image("outputs/figures/ltv_prediction_quality.png", use_container_width=True)
with c2:
    st.subheader("🎯 Recommender Lift vs Random")
    st.image("outputs/figures/recommender_lift.png", use_container_width=True)

st.divider()
st.subheader("📌 Next-Best-Action Distribution Across Customer Base")
st.image("outputs/figures/action_distribution.png", use_container_width=True)
