"""
train_recommender.py
Builds a lightweight collaborative-filtering recommender using Truncated SVD
on the implicit user-category purchase count matrix, then evaluates it with
Precision@K / Recall@K on a held-out most-recent purchase per user.
(Note: this uses scikit-learn's TruncatedSVD rather than the `implicit` ALS
library, to keep the dependency footprint small - the underlying idea of
factorizing a user-item interaction matrix into latent vectors is the same.)
"""
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import TruncatedSVD

K = 3


def main():
    orders = pd.read_csv("data/orders.csv", parse_dates=["order_date"])

    # hold out each user's single most recent order as the test set
    orders = orders.sort_values("order_date")
    last_order_idx = orders.groupby("user_id").tail(1).index
    test_orders = orders.loc[last_order_idx]
    train_orders = orders.drop(index=last_order_idx)

    interaction = train_orders.groupby(["user_id", "product_category"]).size().unstack(fill_value=0)
    user_ids = interaction.index.tolist()
    categories = interaction.columns.tolist()

    n_components = min(6, len(categories) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_latent = svd.fit_transform(interaction.values)
    item_latent = svd.components_.T  # (n_categories, n_components)

    scores = user_latent @ item_latent.T  # (n_users, n_categories) predicted affinity

    recs = {}
    for i, uid in enumerate(user_ids):
        ranked = np.argsort(-scores[i])
        top = [categories[j] for j in ranked[:K]]
        recs[uid] = top

    # ---- Evaluation: Precision@K / Recall@K against held-out most recent purchase ----
    hits, total_evaluated = 0, 0
    for _, row in test_orders.iterrows():
        uid, true_cat = row["user_id"], row["product_category"]
        if uid not in recs:
            continue
        total_evaluated += 1
        if true_cat in recs[uid]:
            hits += 1

    precision_at_k = hits / (total_evaluated * K)
    recall_at_k = hits / total_evaluated if total_evaluated else 0

    # random baseline for context: picking K random categories out of the catalog
    n_categories = len(categories)
    random_recall_at_k = K / n_categories

    metrics = {
        "n_users_evaluated": total_evaluated,
        "k": K,
        "n_categories": n_categories,
        f"precision_at_{K}": round(float(precision_at_k), 4),
        f"recall_at_{K}": round(float(recall_at_k), 4),
        "random_baseline_recall_at_k": round(float(random_recall_at_k), 4),
        "lift_over_random": round(float(recall_at_k / random_recall_at_k), 2) if random_recall_at_k else None,
    }
    with open("outputs/recommender_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    rec_df = pd.DataFrame([{"user_id": uid, "recommended_categories": ", ".join(cats)} for uid, cats in recs.items()])
    rec_df.to_csv("data/recommendations.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
