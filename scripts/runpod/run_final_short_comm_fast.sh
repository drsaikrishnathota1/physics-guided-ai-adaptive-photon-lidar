#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mkdir -p \
  logs \
  results/processed/final_short_comm_fast

echo "Git commit:"
git rev-parse HEAD

echo "Python:"
python --version

echo "Installing dependencies..."
python -m pip install \
  "numpy>=1.26,<3.0" \
  "scipy>=1.12,<2.0" \
  "pandas>=2.2,<3.0" \
  "pyyaml>=6.0,<7.0" \
  "tqdm>=4.66,<5.0" \
  "pytest>=8,<10"

echo "Running focused tests..."
PYTHONPATH=. python -m pytest \
  tests/unit/test_target_absent_episode.py \
  tests/unit/test_policy_ablation.py \
  -q

echo "Running reduced experiment..."
PYTHONPATH=. python \
  scripts/analysis/run_short_comm_final.py \
  --config configs/experiment/final_short_comm_fast.yaml

echo "Creating summaries..."
PYTHONPATH=. python \
  scripts/analysis/summarize_short_comm_final.py \
  --config configs/experiment/final_short_comm_fast.yaml

echo "Outputs:"
find results/processed/final_short_comm_fast \
  -maxdepth 1 \
  -type f \
  -print

echo "Reduced final experiment completed."
