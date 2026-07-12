#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/physics-guided-ai-adaptive-photon-lidar}"

cd "$PROJECT_DIR"

echo "Repository: $PROJECT_DIR"
echo "Commit: $(git rev-parse HEAD)"

python -m pytest \
  tests/unit/test_photon_return_model.py \
  tests/integration/test_physics_validation.py \
  -v

python scripts/analysis/validate_photon_physics.py

echo "Generated validation files:"
find results/processed/physics_validation -maxdepth 1 -type f -print
find figures/simulation -maxdepth 1 -type f -print

echo "Physics validation completed successfully."
