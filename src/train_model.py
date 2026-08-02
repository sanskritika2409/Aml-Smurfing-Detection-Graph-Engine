"""
train_model.py
Trains two gradient-boosted models sharing the same feature set:
  - Churn classifier (XGBoost, binary) 
  - LTV regressor (XGBoost, log-target regression)
Both are trained together in this script to mirror a shared feature pipeline
(a lightweight, practical alternative to a full multi-task neural net), then
evaluated with churn AUC/F1 and LTV MAPE.
"""
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_percentage_error, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

NUMERIC_FEATURES = [
    "account_age_days", "recency_days", "frequency_7d", "frequency_30d",
    "monetary_30d", "monetary_total", "avg_session_duration", "recent_avg_duration",
    "avg_pages_viewed", "login_trend_slope", "n_orders_total", "avg_order_value",
]
CATEGORICAL_FEATURES = ["plan_type", "region", "top_category"]


def encode_categoricals(df, encoders=None, fit=True):
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_FEATURES:
        if fit:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col + "_enc"] = df[col].astype(str).map(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
    return df, encoders


def main():
    df = pd.read_csv("data/customer_features.csv")
    df, encoders = encode_categoricals(df, fit=True)

    feature_cols = NUMERIC_FEATURES + [c + "_enc" for c in CATEGORICAL_FEATURES]
    X = df[feature_cols]
    y_churn = df["churn_label"]
    y_ltv = np.log1p(df["ltv_12m"])

    X_train, X_test, ychurn_train, ychurn_test, yltv_train, yltv_test = train_test_split(
        X, y_churn, y_ltv, test_size=0.2, random_state=42, stratify=y_churn
    )

    # ---- Churn classifier ----
    churn_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42,
    )
    churn_model.fit(X_train, ychurn_train)
    churn_pred_proba = churn_model.predict_proba(X_test)[:, 1]
    churn_pred = (churn_pred_proba >= 0.5).astype(int)

    churn_metrics = {
        "auc_roc": round(float(roc_auc_score(ychurn_test, churn_pred_proba)), 4),
        "f1_score": round(float(f1_score(ychurn_test, churn_pred)), 4),
        "precision": round(float(precision_score(ychurn_test, churn_pred)), 4),
        "recall": round(float(recall_score(ychurn_test, churn_pred)), 4),
    }

    # ---- LTV regressor ----
    ltv_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    ltv_model.fit(X_train, yltv_train)
    ltv_pred_log = ltv_model.predict(X_test)
    ltv_pred = np.expm1(ltv_pred_log)
    ltv_actual = np.expm1(yltv_test)

    # MAPE on customers with non-trivial actual LTV (avoids divide-by-near-zero blowups, standard practice)
    mask = ltv_actual > 10
    mape = mean_absolute_percentage_error(ltv_actual[mask], np.clip(ltv_pred[mask], 0, None))

    ltv_metrics = {
        "mape_on_active_customers": round(float(mape), 4),
        "mean_actual_ltv": round(float(ltv_actual.mean()), 2),
        "mean_predicted_ltv": round(float(ltv_pred.mean()), 2),
    }

    metrics = {"churn_model": churn_metrics, "ltv_model": ltv_metrics}
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # feature importances for interpretability
    churn_importance = pd.Series(churn_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    ltv_importance = pd.Series(ltv_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    churn_importance.to_csv("outputs/churn_feature_importance.csv")
    ltv_importance.to_csv("outputs/ltv_feature_importance.csv")

    joblib.dump(
        {"churn_model": churn_model, "ltv_model": ltv_model, "encoders": encoders, "feature_cols": feature_cols},
        "outputs/digital_twin_model.joblib",
    )

    # score the full dataset for downstream API/dashboard use
    df["churn_score"] = churn_model.predict_proba(X)[:, 1]
    df["predicted_ltv_12m"] = np.expm1(ltv_model.predict(X))
    df.to_csv("data/customer_scored.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
