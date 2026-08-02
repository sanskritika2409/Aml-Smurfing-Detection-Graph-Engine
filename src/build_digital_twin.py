"""
build_digital_twin.py
Combines the churn score, predicted LTV, and category recommendations into a
single "Digital Twin" record per customer, with a business-rule orchestrator
that maps the combination to a concrete next-best-action.
"""
import pandas as pd
import json

HIGH_LTV_THRESHOLD_PERCENTILE = 0.6  # top 40% of predicted LTV counts as "high value"


def decide_action(churn_score, ltv_percentile, high_churn=0.6, low_churn=0.3):
    if churn_score >= high_churn and ltv_percentile >= HIGH_LTV_THRESHOLD_PERCENTILE:
        return "RETENTION_OFFER", "Send a 20-30% discount coupon before they leave - high value, high churn risk."
    elif churn_score <= low_churn and ltv_percentile >= HIGH_LTV_THRESHOLD_PERCENTILE:
        return "UPSELL", "Recommend Premium tier / bundle - engaged and high value."
    elif churn_score >= high_churn and ltv_percentile < HIGH_LTV_THRESHOLD_PERCENTILE:
        return "LOW_COST_NURTURE", "Send an automated re-engagement email - at risk but low value, don't over-invest."
    else:
        return "STANDARD_NEWSLETTER", "No urgent action - continue standard lifecycle messaging."


def main():
    df = pd.read_csv("data/customer_scored.csv")
    recs = pd.read_csv("data/recommendations.csv")
    df = df.merge(recs, on="user_id", how="left")
    df["recommended_categories"] = df["recommended_categories"].fillna("Grocery, Electronics, Fashion")

    df["ltv_percentile"] = df["predicted_ltv_12m"].rank(pct=True)

    actions, messages = [], []
    for _, r in df.iterrows():
        action, message = decide_action(r["churn_score"], r["ltv_percentile"])
        actions.append(action)
        messages.append(message)
    df["system_action"] = actions
    df["action_message"] = messages

    df.to_csv("data/digital_twin.csv", index=False)

    action_summary = df["system_action"].value_counts().to_dict()
    with open("outputs/action_distribution.json", "w") as f:
        json.dump(action_summary, f, indent=2)

    print(json.dumps(action_summary, indent=2))


if __name__ == "__main__":
    main()
