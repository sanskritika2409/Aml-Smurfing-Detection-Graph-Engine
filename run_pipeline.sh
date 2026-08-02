#!/usr/bin/env bash
# Reproduces the full pipeline end-to-end
set -e
echo "1/6 Generating synthetic customer, session, and order data..."
python3 src/generate_data.py

echo "2/6 Engineering RFM + behavioral features..."
python3 src/build_features.py

echo "3/6 Training churn (XGBoost) and LTV (XGBoost) models..."
python3 src/train_model.py

echo "4/6 Training SVD-based recommender..."
python3 src/train_recommender.py

echo "5/6 Building the unified Digital Twin + business-rule orchestrator..."
python3 src/build_digital_twin.py

echo "6/6 Generating charts..."
python3 src/make_visuals.py

echo "Done. See outputs/figures/ for charts and data/digital_twin.csv for unified customer records."
