#!/usr/bin/env bash

set -euo pipefail

echo "=== Repository preflight ==="

required_files=(
  "README.md"
  "requirements.txt"
  "configs/base.yaml"
  "src/physics/photon_return_model.py"
  "scripts/analysis/validate_photon_physics.py"
  "scripts/runpod/run_physics_validation.sh"
  "tests/unit/test_photon_return_model.py"
  "tests/integration/test_physics_validation.py"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file"
    exit 1
  fi
done

echo "Required files: OK"

python3 -m compileall -q src scripts tests

echo "Python syntax: OK"

git diff --check

echo "Git whitespace check: OK"

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse HEAD)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Warning: working tree has uncommitted changes"
  git status --short
else
  echo "Working tree: clean"
fi

echo "Preflight completed successfully."
