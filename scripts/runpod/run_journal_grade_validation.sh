#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== Journal-grade physics validation ==="
echo "Commit: $(git rev-parse HEAD)"

python -m pip install --upgrade pip
python -m pip install \
  "numpy>=1.26,<3.0" \
  "pytest>=8,<10"

python -m pytest \
  tests/unit/test_photon_return_model.py \
  tests/unit/test_pulse_accounting.py \
  tests/unit/test_beam_geometry.py \
  tests/unit/test_atmospheric_channel.py \
  tests/unit/test_detector_nonidealities.py \
  tests/integration/test_physics_validation.py \
  tests/integration/test_journal_grade_model.py \
  -v

python scripts/analysis/validate_journal_grade_model.py

echo "=== Output ==="
cat results/processed/journal_grade_validation/summary.json

echo "Journal-grade physics validation completed."
