import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc

plt.style.use("seaborn-v0_8-darkgrid")

df = pd.read_csv("data/digital_twin.csv")

with open("outputs/metrics.json") as f:
    metrics = json.load(f)
with open("outputs/recommender_metrics.json") as f:
    rec_metrics = json.load(f)

# ---- 1. Churn score distribution: churners vs non-churners ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df[df["churn_label"] == 1]["churn_score"], bins=30, alpha=0.65, label="Actually churned", color="#DE350B")
ax.hist(df[df["churn_label"] == 0]["churn_score"], bins=30, alpha=0.65, label="Actually retained", color="#00875A")
ax.set_xlabel("Predicted Churn Score")
ax.set_ylabel("Number of Customers")
ax.set_title(f"Churn Score Separation (AUC-ROC = {metrics['churn_model']['auc_roc']})")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/churn_score_distribution.png", dpi=150)
plt.close()

# ---- 2. ROC curve ----
fpr, tpr, _ = roc_curve(df["churn_label"], df["churn_score"])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="#0052CC", lw=2, label=f"XGBoost (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Churn Classifier ROC Curve")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/churn_roc_curve.png", dpi=150)
plt.close()

# ---- 3. Predicted vs Actual LTV scatter ----
fig, ax = plt.subplots(figsize=(7, 6))
sample = df.sample(min(1500, len(df)), random_state=42)
ax.scatter(sample["ltv_12m"], sample["predicted_ltv_12m"], alpha=0.3, s=15, color="#6554C0")
max_val = max(sample["ltv_12m"].max(), sample["predicted_ltv_12m"].max())
ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=1, label="Perfect prediction")
ax.set_xlabel("Actual 12-Month LTV ($)")
ax.set_ylabel("Predicted 12-Month LTV ($)")
ax.set_title(f"LTV Prediction Quality (MAPE = {metrics['ltv_model']['mape_on_active_customers']:.1%})")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/ltv_prediction_quality.png", dpi=150)
plt.close()

# ---- 4. Feature importance (churn model) ----
churn_imp = pd.read_csv("outputs/churn_feature_importance.csv", index_col=0).squeeze("columns").head(10)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(churn_imp.index[::-1], churn_imp.values[::-1], color="#FF8B00")
ax.set_xlabel("Feature Importance (XGBoost gain)")
ax.set_title("Top 10 Churn Drivers")
plt.tight_layout()
plt.savefig("outputs/figures/churn_feature_importance.png", dpi=150)
plt.close()

# ---- 5. Recommender: lift over random baseline ----
fig, ax = plt.subplots(figsize=(6, 5))
labels = ["Random Baseline", "SVD Collaborative\nFiltering (Ours)"]
values = [rec_metrics["random_baseline_recall_at_k"], rec_metrics[f"recall_at_{rec_metrics['k']}"]]
colors_bar = ["#8993A4", "#00875A"]
bars = ax.bar(labels, values, color=colors_bar)
ax.set_ylabel(f"Recall@{rec_metrics['k']}")
ax.set_ylim(0, 1)
ax.set_title(f"Recommender Lift: {rec_metrics['lift_over_random']}x over random")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center")
plt.tight_layout()
plt.savefig("outputs/figures/recommender_lift.png", dpi=150)
plt.close()

# ---- 6. System action distribution (business orchestrator output) ----
action_counts = df["system_action"].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
colors_map = {"RETENTION_OFFER": "#DE350B", "UPSELL": "#00875A", "LOW_COST_NURTURE": "#FFAB00", "STANDARD_NEWSLETTER": "#4C9AFF"}
ax.bar(action_counts.index, action_counts.values, color=[colors_map.get(a, "#8993A4") for a in action_counts.index])
ax.set_ylabel("Number of Customers")
ax.set_title("Digital Twin: Recommended Next-Best-Action Distribution")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/figures/action_distribution.png", dpi=150)
plt.close()

print("All Project 2 visuals generated in outputs/figures/")
