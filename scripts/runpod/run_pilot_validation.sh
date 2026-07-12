#!/usr/bin/env bash

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== RunPod pilot validation ==="
echo "Commit: $(git rev-parse HEAD)"

python -m pip install --upgrade pip

python -m pip install \
  numpy \
  scipy \
  pandas \
  matplotlib \
  pytest

python -m pytest \
  tests/unit/test_photon_return_model.py \
  tests/integration/test_physics_validation.py \
  -v

python scripts/analysis/validate_photon_physics.py

echo "=== Generated outputs ==="
find results/processed/physics_validation -maxdepth 1 -type f -print
find figures/simulation -maxdepth 1 -type f -print

echo "Pilot validation completed."
