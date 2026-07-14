#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p \
  results/processed/baseline_comparison \
  reports/runpod/baseline_comparison

echo "=== Baseline photon-allocation comparison ==="
echo "Commit: $(git rev-parse HEAD)"

git rev-parse HEAD \
  > reports/runpod/baseline_comparison/source_commit.txt

python --version \
  > reports/runpod/baseline_comparison/python_version.txt

python -m pip install --upgrade pip

python -m pip install \
  "numpy>=1.26,<3.0" \
  "pytest>=8,<10"

python -m pip freeze \
  > reports/runpod/baseline_comparison/pip_freeze.txt

python -m pytest \
  tests/unit/test_adaptive_lidar_env.py \
  tests/unit/test_baseline_policies.py \
  tests/unit/test_pulse_accounting.py \
  tests/unit/test_beam_geometry.py \
  tests/unit/test_atmospheric_channel.py \
  tests/unit/test_detector_nonidealities.py \
  tests/integration/test_journal_grade_model.py \
  -v

python scripts/analysis/compare_baseline_policies.py

echo "=== Baseline summary ==="
cat results/processed/baseline_comparison/summary.json

echo "Baseline comparison completed."
