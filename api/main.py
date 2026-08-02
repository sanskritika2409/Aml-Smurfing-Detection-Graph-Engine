"""
FastAPI serving layer for the AML Smurfing Detection Engine.
Run with: uvicorn api.main:app --reload --port 8000
Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException
import pandas as pd
import networkx as nx

app = FastAPI(
    title="AML Smurfing Detection Engine",
    description="Graph-based unsupervised anomaly detection API for identifying structured money-laundering (smurfing) rings.",
    version="1.0.0",
)

DF = pd.read_csv("data/account_risk_scores_final.csv")
G = nx.read_gexf("data/transaction_graph.gexf")


@app.get("/")
def root():
    return {"service": "aml-smurfing-detection-engine", "status": "running", "accounts_indexed": len(DF)}


@app.get("/aml/v1/screen/{account_id}")
def screen_account(account_id: str):
    row = DF[DF["account_id"] == account_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Account not found")
    r = row.iloc[0]

    neighbors = []
    if account_id in G:
        neighbors = list(set(G.predecessors(account_id)) | set(G.successors(account_id)))

    return {
        "account_id": account_id,
        "risk_score": round(float(r["risk_score"]), 4),
        "risk_percentile": round(float(r["risk_percentile"]), 2),
        "alert_status": r["alert_status"],
        "final_alert": bool(r["final_alert"]),
        "linked_accounts": neighbors[:20],
        "n_linked_accounts": len(neighbors),
    }


@app.get("/aml/v1/top-alerts")
def top_alerts(limit: int = 20):
    top = DF.sort_values("risk_score", ascending=False).head(limit)
    return top[["account_id", "role", "risk_score", "risk_percentile", "alert_status", "final_alert"]].to_dict(
        orient="records"
    )


@app.get("/aml/v1/metrics")
def metrics():
    import json
    with open("outputs/metrics_after_propagation.json") as f:
        return json.load(f)
