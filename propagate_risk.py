"""
propagate_risk.py
Node-level anomaly scores alone catch "hub" accounts (kingpins/sinks) very well,
but individual mule accounts (a single mid-sized transfer) look statistically
close to normal customers. This step propagates risk across the graph: any
account transacting directly with a CRITICAL-risk hub inherits an elevated
"network risk" score. This mirrors how real AML investigators expand a case
from a seed alert to its immediate transaction neighborhood.
"""
import pandas as pd
import networkx as nx
import json

TOP_SEED_PERCENTILE = 99  # accounts above this percentile become "seeds"


def main():
    G = nx.read_gexf("data/transaction_graph.gexf")
    df = pd.read_csv("data/account_risk_scores.csv")

    seed_threshold = df["risk_score"].quantile(TOP_SEED_PERCENTILE / 100)
    seeds = set(df.loc[df["risk_score"] >= seed_threshold, "account_id"])

    neighbor_risk = {}
    for seed in seeds:
        if seed not in G:
            continue
        for nbr in set(G.predecessors(seed)) | set(G.successors(seed)):
            neighbor_risk[nbr] = max(neighbor_risk.get(nbr, 0), 1)

    df["is_seed"] = df["account_id"].isin(seeds)
    df["network_flag"] = df["account_id"].map(neighbor_risk).fillna(0)
    df["final_alert"] = df["is_seed"] | (df["network_flag"] == 1)

    ground_truth = df["role"].isin(["kingpin", "mule", "sink"])
    n_true = ground_truth.sum()
    caught = (df["final_alert"] & ground_truth).sum()
    total_flagged = df["final_alert"].sum()

    metrics = {
        "seed_accounts_flagged": int(len(seeds)),
        "total_accounts_flagged_after_propagation": int(total_flagged),
        "ground_truth_smurf_accounts": int(n_true),
        "recall_after_propagation": round(float(caught / n_true), 4),
        "precision_after_propagation": round(float(caught / total_flagged), 4),
    }

    with open("outputs/metrics_after_propagation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    df.to_csv("data/account_risk_scores_final.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
