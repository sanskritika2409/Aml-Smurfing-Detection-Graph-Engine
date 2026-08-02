import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from pyvis.network import Network

plt.style.use("seaborn-v0_8-darkgrid")

df = pd.read_csv("data/account_risk_scores_final.csv")
G = nx.read_gexf("data/transaction_graph.gexf")

# ---- 1. Risk score distribution by role ----
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"normal": "#4C9AFF", "mule": "#FFAB00", "kingpin": "#DE350B", "sink": "#6554C0"}
for role, color in colors.items():
    subset = df[df["role"] == role]["risk_score"]
    ax.hist(subset, bins=30, alpha=0.6, label=role, color=color)
ax.set_xlabel("Risk Score")
ax.set_ylabel("Number of Accounts")
ax.set_title("Anomaly Risk Score Distribution by Account Role")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/risk_score_distribution.png", dpi=150)
plt.close()

# ---- 2. Precision/Recall before vs after propagation ----
with open("outputs/metrics.json") as f:
    m1 = json.load(f)
with open("outputs/metrics_after_propagation.json") as f:
    m2 = json.load(f)

labels = ["Recall", "Precision"]
before = [m1["recall_at_352"], m1["precision_at_352"]]
after = [m2["recall_after_propagation"], m2["precision_after_propagation"]]

fig, ax = plt.subplots(figsize=(7, 5))
x = range(len(labels))
w = 0.35
ax.bar([i - w / 2 for i in x], before, width=w, label="Node features only", color="#8993A4")
ax.bar([i + w / 2 for i in x], after, width=w, label="+ Graph risk propagation", color="#00875A")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.set_title("Detection Quality: Node Features vs Graph Propagation")
ax.legend()
for i, v in enumerate(before):
    ax.text(i - w / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
for i, v in enumerate(after):
    ax.text(i + w / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/figures/precision_recall_comparison.png", dpi=150)
plt.close()

# ---- 3. Feature importance proxy (Isolation Forest doesn't give direct importances;
#         use mean absolute deviation contribution as an interpretable proxy) ----
feature_cols = [
    "in_degree", "out_degree", "degree_ratio", "pagerank", "betweenness",
    "clustering_coeff", "total_sent", "total_received", "n_txn_sent",
    "n_txn_received", "unique_receivers", "unique_senders", "avg_sent_amount",
    "std_sent_amount", "round_amount_ratio", "max_hourly_velocity", "device_diversity",
]
top_flagged = df[df["final_alert"]][feature_cols]
normal = df[df["role"] == "normal"][feature_cols]
diff = ((top_flagged.mean() - normal.mean()).abs() / (normal.std() + 1e-9)).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(diff.index[::-1], diff.values[::-1], color="#0052CC")
ax.set_xlabel("Standardized Separation (flagged vs normal accounts)")
ax.set_title("Top Discriminative Features for Smurfing Detection")
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance.png", dpi=150)
plt.close()

# ---- 4. Static network snapshot of one ring + surrounding normal accounts ----
ring_nodes = set(df[df["account_id"].str.contains("KING_00|MULE_00|SINK_00", regex=True, na=False)]["account_id"])
context_nodes = set()
for n in ring_nodes:
    if n in G:
        context_nodes.update(G.predecessors(n))
        context_nodes.update(G.successors(n))
subG = G.subgraph(ring_nodes | context_nodes).copy()

role_map = dict(zip(df["account_id"], df["role"]))
node_colors = [colors.get(role_map.get(n, "normal"), "#4C9AFF") for n in subG.nodes()]
node_sizes = [700 if role_map.get(n, "normal") in ("kingpin", "sink") else 120 for n in subG.nodes()]

fig, ax = plt.subplots(figsize=(9, 7))
pos = nx.spring_layout(subG, seed=42, k=0.4)
nx.draw_networkx_edges(subG, pos, alpha=0.3, arrows=True, arrowsize=6, ax=ax)
nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
patches = [mpatches.Patch(color=c, label=r) for r, c in colors.items()]
ax.legend(handles=patches, loc="upper right")
ax.set_title("Smurfing Ring #00: Kingpin -> Mules -> Offshore Sink")
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/figures/smurfing_ring_network.png", dpi=150)
plt.close()

# ---- 5. Interactive Pyvis HTML graph (open in browser) ----
net = Network(height="750px", width="100%", directed=True, notebook=False, bgcolor="#111827", font_color="white")
for n in subG.nodes():
    role = role_map.get(n, "normal")
    net.add_node(n, label=n, color=colors.get(role, "#4C9AFF"),
                 size=30 if role in ("kingpin", "sink") else 12,
                 title=f"role: {role}")
for u, v, data in subG.edges(data=True):
    net.add_edge(u, v, value=data.get("weight", 1), title=f"${data.get('weight', 0):,.2f}")
net.write_html("outputs/figures/interactive_smurfing_graph.html", open_browser=False, notebook=False)

print("All visuals generated in outputs/figures/")
