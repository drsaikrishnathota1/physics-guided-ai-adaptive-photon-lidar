#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mkdir -p \
  logs \
  results/processed/final_short_comm

echo "Git commit:"
git rev-parse HEAD

echo "Python:"
python --version

echo "Installing required packages..."
python -m pip install \
  --upgrade \
  numpy \
  scipy \
  pandas \
  pyyaml \
  tqdm \
  pytest

echo "Running unit tests..."
python -m pytest \
  tests/unit/test_target_absent_episode.py \
  tests/unit/test_policy_ablation.py \
  -q

echo "Running smoke test..."
python scripts/analysis/run_short_comm_final.py \
  --smoke-test

echo "Smoke-test summary..."
python scripts/analysis/summarize_short_comm_final.py

rm -rf results/processed/final_short_comm

echo "Running complete final experiment..."
python scripts/analysis/run_short_comm_final.py

echo "Creating statistical summaries..."
python scripts/analysis/summarize_short_comm_final.py

echo "Final outputs:"
find results/processed/final_short_comm \
  -maxdepth 1 \
  -type f \
  -print

echo "Final short-communication experiment completed."
