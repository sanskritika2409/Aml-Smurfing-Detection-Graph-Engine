"""
build_features.py
Rolls up raw clickstream and order logs into a per-user feature snapshot:
RFM scores, engagement trend (login slope), session decay, and product affinity.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

END_DATE = datetime(2025, 12, 31)
FEATURE_CUTOFF = END_DATE - timedelta(days=30)  # features computed on data BEFORE the label window


def compute_login_slope(session_dates, cutoff):
    """Simple week-over-week trend: negative slope = declining engagement."""
    if len(session_dates) < 2:
        return 0.0
    weeks = pd.Series(session_dates).apply(lambda d: (cutoff - d).days // 7)
    weekly_counts = weeks.value_counts().sort_index(ascending=False)  # week 0 = most recent
    if len(weekly_counts) < 2:
        return 0.0
    x = np.arange(len(weekly_counts))
    y = weekly_counts.values
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def main():
    users = pd.read_csv("data/users.csv", parse_dates=["signup_date"])
    sessions = pd.read_csv("data/sessions.csv", parse_dates=["session_date"])
    orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])

    # only use activity BEFORE the cutoff to build features (avoid label leakage)
    sess_feat = sessions[sessions["session_date"] < FEATURE_CUTOFF]
    ord_feat = orders[orders["order_date"] < FEATURE_CUTOFF]

    rows = []
    sess_by_user = sess_feat.groupby("user_id")
    ord_by_user = ord_feat.groupby("user_id")

    for _, u in users.iterrows():
        uid = u["user_id"]
        u_sessions = sess_by_user.get_group(uid) if uid in sess_by_user.groups else pd.DataFrame(columns=sessions.columns)
        u_orders = ord_by_user.get_group(uid) if uid in ord_by_user.groups else pd.DataFrame(columns=orders.columns)

        recency_days = (FEATURE_CUTOFF - u_sessions["session_date"].max()).days if len(u_sessions) else 999
        frequency_7d = (u_sessions["session_date"] >= FEATURE_CUTOFF - timedelta(days=7)).sum()
        frequency_30d = (u_sessions["session_date"] >= FEATURE_CUTOFF - timedelta(days=30)).sum()
        monetary_30d = u_orders.loc[u_orders["order_date"] >= FEATURE_CUTOFF - timedelta(days=30), "revenue"].sum()
        monetary_total = u_orders["revenue"].sum()

        avg_session_duration = u_sessions["duration_minutes"].mean() if len(u_sessions) else 0
        recent_avg_duration = u_sessions.loc[u_sessions["session_date"] >= FEATURE_CUTOFF - timedelta(days=14), "duration_minutes"].mean()
        recent_avg_duration = 0 if pd.isna(recent_avg_duration) else recent_avg_duration
        avg_pages = u_sessions["pages_viewed"].mean() if len(u_sessions) else 0

        login_slope = compute_login_slope(u_sessions["session_date"].tolist(), FEATURE_CUTOFF)

        n_orders_total = len(u_orders)
        avg_order_value = u_orders["revenue"].mean() if len(u_orders) else 0
        top_category = u_orders["product_category"].mode()[0] if len(u_orders) else "None"

        account_age_days = (FEATURE_CUTOFF - pd.Timestamp(u["signup_date"])).days

        rows.append({
            "user_id": uid,
            "plan_type": u["plan_type"],
            "region": u["region"],
            "account_age_days": account_age_days,
            "recency_days": recency_days,
            "frequency_7d": frequency_7d,
            "frequency_30d": frequency_30d,
            "monetary_30d": monetary_30d,
            "monetary_total": monetary_total,
            "avg_session_duration": avg_session_duration,
            "recent_avg_duration": recent_avg_duration,
            "avg_pages_viewed": avg_pages,
            "login_trend_slope": login_slope,
            "n_orders_total": n_orders_total,
            "avg_order_value": avg_order_value,
            "top_category": top_category,
            "churn_label": u["churn_label"],
        })

    features = pd.DataFrame(rows).fillna(0)

    # 12-month LTV target: total revenue extrapolated from observed monetary rate (regression target).
    # Account age is floored at 30 days to avoid extreme extrapolation for brand-new accounts,
    # and the result is capped at the 99th percentile to control outlier influence (standard LTV modeling practice).
    raw_ltv = (features["monetary_total"] / features["account_age_days"].clip(lower=30)) * 365
    cap = raw_ltv.quantile(0.99)
    features["ltv_12m"] = raw_ltv.clip(upper=cap)

    features.to_csv("data/customer_features.csv", index=False)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Churn rate: {features['churn_label'].mean():.1%}")
    print(features[["recency_days", "frequency_30d", "monetary_30d", "ltv_12m"]].describe())


if __name__ == "__main__":
    main()
