#!/usr/bin/env bash
# Reproduces the full pipeline end-to-end: data -> features -> model -> propagation -> visuals
set -e
echo "1/5 Generating synthetic transaction data..."
python3 src/generate_data.py

echo "2/5 Building transaction graph and engineering features..."
python3 src/build_graph_features.py

echo "3/5 Training unsupervised anomaly ensemble..."
python3 src/train_model.py

echo "4/5 Running graph-based risk propagation..."
python3 src/propagate_risk.py

echo "5/5 Generating charts and network visuals..."
python3 src/make_visuals.py

echo "Done. See outputs/figures/ for charts and data/account_risk_scores_final.csv for results."
