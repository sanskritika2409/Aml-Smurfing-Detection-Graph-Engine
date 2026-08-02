"""
build_graph_features.py
Builds a directed transaction graph with NetworkX and computes topological +
behavioral features per account, used later for unsupervised anomaly detection.
"""
import pandas as pd
import numpy as np
import networkx as nx


def build_graph(txns: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in txns.iterrows():
        s, r, amt = row["sender_account"], row["receiver_account"], row["amount"]
        if G.has_edge(s, r):
            G[s][r]["weight"] += amt
            G[s][r]["count"] += 1
        else:
            G.add_edge(s, r, weight=amt, count=1)
    return G


def compute_features(txns: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    accounts = list(G.nodes())

    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, k=min(500, len(G)), weight="weight", seed=42)
    clustering = nx.clustering(G.to_undirected())

    txns = txns.copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    txns["is_round"] = (txns["amount"] % 1 == 0).astype(int)

    sent = txns.groupby("sender_account")
    recv = txns.groupby("receiver_account")

    total_sent = sent["amount"].sum()
    total_recv = recv["amount"].sum()
    n_sent = sent["amount"].count()
    n_recv = recv["amount"].count()
    unique_receivers = sent["receiver_account"].nunique()
    unique_senders = recv["sender_account"].nunique()
    avg_sent_amt = sent["amount"].mean()
    std_sent_amt = sent["amount"].std().fillna(0)
    round_ratio = sent["is_round"].mean()

    # velocity: max transactions sent by an account within any 1-hour rolling window
    velocity = {}
    for acc, grp in sent:
        ts = grp["timestamp"].sort_values()
        if len(ts) < 2:
            velocity[acc] = 0
            continue
        counts = []
        ts_arr = ts.values
        for t in ts_arr:
            window = ts_arr[(ts_arr >= t) & (ts_arr <= t + np.timedelta64(1, "h"))]
            counts.append(len(window))
        velocity[acc] = max(counts)

    # unique devices used per sender (device reuse across many accounts is a red flag upstream,
    # here we track how many distinct devices an account itself used)
    device_count = sent["device_id"].nunique() if "device_id" in txns.columns else pd.Series(dtype=int)

    rows = []
    for acc in accounts:
        rows.append({
            "account_id": acc,
            "in_degree": in_degree.get(acc, 0),
            "out_degree": out_degree.get(acc, 0),
            "degree_ratio": (out_degree.get(acc, 0) + 1) / (in_degree.get(acc, 0) + 1),
            "pagerank": pagerank.get(acc, 0),
            "betweenness": betweenness.get(acc, 0),
            "clustering_coeff": clustering.get(acc, 0),
            "total_sent": total_sent.get(acc, 0),
            "total_received": total_recv.get(acc, 0),
            "n_txn_sent": n_sent.get(acc, 0),
            "n_txn_received": n_recv.get(acc, 0),
            "unique_receivers": unique_receivers.get(acc, 0),
            "unique_senders": unique_senders.get(acc, 0),
            "avg_sent_amount": avg_sent_amt.get(acc, 0),
            "std_sent_amount": std_sent_amt.get(acc, 0),
            "round_amount_ratio": round_ratio.get(acc, 0),
            "max_hourly_velocity": velocity.get(acc, 0),
            "device_diversity": device_count.get(acc, 0) if len(device_count) else 0,
        })
    return pd.DataFrame(rows).fillna(0)


def main():
    txns = pd.read_csv("data/transactions.csv")
    accounts = pd.read_csv("data/accounts.csv")

    G = build_graph(txns)
    features = compute_features(txns, G)
    features = features.merge(accounts[["account_id", "is_smurf_related", "role"]], on="account_id", how="left")
    features["is_smurf_related"] = features["is_smurf_related"].fillna(False)
    features["role"] = features["role"].fillna("normal")

    features.to_csv("data/account_features.csv", index=False)
    nx.write_gexf(G, "data/transaction_graph.gexf")
    print(f"Feature matrix shape: {features.shape}")
    print(features["role"].value_counts())


if __name__ == "__main__":
    main()
