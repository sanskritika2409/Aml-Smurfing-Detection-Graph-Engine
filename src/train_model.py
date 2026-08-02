"""
train_model.py
Trains an unsupervised anomaly ensemble (Isolation Forest + a simple
reconstruction-based detector using PCA as a lightweight autoencoder proxy)
on the graph/behavioral features, then evaluates detection quality against
the ground-truth smurf labels (used ONLY for evaluation, never for training).
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_COLS = [
    "in_degree", "out_degree", "degree_ratio", "pagerank", "betweenness",
    "clustering_coeff", "total_sent", "total_received", "n_txn_sent",
    "n_txn_received", "unique_receivers", "unique_senders", "avg_sent_amount",
    "std_sent_amount", "round_amount_ratio", "max_hourly_velocity", "device_diversity",
]


def main():
    df = pd.read_csv("data/account_features.csv")
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Model A: Isolation Forest
    iso = IsolationForest(n_estimators=300, contamination=0.15, random_state=42)
    iso.fit(Xs)
    iso_scores = -iso.score_samples(Xs)  # higher = more anomalous

    # Model B: PCA reconstruction error (lightweight autoencoder proxy -
    # compresses to a low-dim latent space and reconstructs; accounts with
    # unusual structural patterns reconstruct poorly, same idea as a neural autoencoder)
    pca = PCA(n_components=6, random_state=42)
    latent = pca.fit_transform(Xs)
    reconstructed = pca.inverse_transform(latent)
    recon_error = np.mean((Xs - reconstructed) ** 2, axis=1)

    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    iso_norm = normalize(iso_scores)
    recon_norm = normalize(recon_error)

    df["iso_forest_score"] = iso_norm
    df["reconstruction_score"] = recon_norm
    df["risk_score"] = 0.6 * iso_norm + 0.4 * recon_norm
    df["risk_percentile"] = df["risk_score"].rank(pct=True) * 100

    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df["alert_status"] = np.select(
        [df["risk_percentile"] >= 95, df["risk_percentile"] >= 85],
        ["CRITICAL", "WARNING"],
        default="NORMAL",
    )

    # ---- Evaluation vs ground truth (labels only used here, never for training) ----
    ground_truth = df["role"].isin(["kingpin", "mule", "sink"])
    n_true_positive_accounts = ground_truth.sum()

    top_k = int(n_true_positive_accounts)
    top_k_flagged = df.head(top_k)
    recall_at_k = top_k_flagged["role"].isin(["kingpin", "mule", "sink"]).sum() / n_true_positive_accounts

    top_100 = df.head(100)
    recall_at_100 = top_100["role"].isin(["kingpin", "mule", "sink"]).sum() / min(100, n_true_positive_accounts)

    precision_at_k = top_k_flagged["role"].isin(["kingpin", "mule", "sink"]).sum() / top_k

    metrics = {
        "n_accounts": int(len(df)),
        "n_ground_truth_smurf_accounts": int(n_true_positive_accounts),
        f"recall_at_{top_k}": round(float(recall_at_k), 4),
        "recall_at_100": round(float(recall_at_100), 4),
        f"precision_at_{top_k}": round(float(precision_at_k), 4),
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df.to_csv("data/account_risk_scores.csv", index=False)
    joblib.dump({"scaler": scaler, "iso_forest": iso, "pca": pca, "feature_cols": FEATURE_COLS},
                "outputs/aml_model.joblib")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
