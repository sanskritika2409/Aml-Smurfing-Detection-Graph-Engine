"""
FastAPI serving layer for the Customer Digital Twin Engine.
Run with: uvicorn api.main:app --reload --port 8001
Then visit http://127.0.0.1:8001/docs for interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI(
    title="Customer Digital Twin Engine",
    description="Unified churn, LTV, and recommendation API returning a single next-best-action per customer.",
    version="1.0.0",
)

DF = pd.read_csv("data/digital_twin.csv")


@app.get("/")
def root():
    return {"service": "customer-digital-twin-engine", "status": "running", "customers_indexed": len(DF)}


@app.get("/customer/v1/twin/{user_id}")
def get_twin(user_id: str):
    row = DF[DF["user_id"] == user_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    r = row.iloc[0]
    return {
        "customer_id": user_id,
        "digital_twin": {
            "churn_score": round(float(r["churn_score"]), 4),
            "predicted_ltv_12m": round(float(r["predicted_ltv_12m"]), 2),
            "ltv_percentile": round(float(r["ltv_percentile"]), 3),
            "recommended_categories": r["recommended_categories"].split(", "),
            "system_action": r["system_action"],
            "action_message": r["action_message"],
        },
    }


@app.get("/customer/v1/segment/{action}")
def get_segment(action: str, limit: int = 25):
    valid = {"RETENTION_OFFER", "UPSELL", "LOW_COST_NURTURE", "STANDARD_NEWSLETTER"}
    if action not in valid:
        raise HTTPException(status_code=400, detail=f"action must be one of {valid}")
    subset = DF[DF["system_action"] == action].head(limit)
    return subset[["user_id", "churn_score", "predicted_ltv_12m", "system_action"]].to_dict(orient="records")


@app.get("/customer/v1/metrics")
def metrics():
    import json
    with open("outputs/metrics.json") as f:
        m = json.load(f)
    with open("outputs/recommender_metrics.json") as f:
        m["recommender"] = json.load(f)
    return m
