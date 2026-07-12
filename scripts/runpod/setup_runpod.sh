#!/usr/bin/env bash

set -euo pipefail

echo "Setting up Physics-Guided Adaptive Photon LiDAR project"

python --version
nvidia-smi || true

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p \
  data/raw \
  data/interim \
  data/processed \
  models/checkpoints \
  results/raw \
  results/processed \
  reports/runpod

python - <<'PY'
import platform
import sys

import numpy
import torch

print("Python:", sys.version)
print("Platform:", platform.platform())
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY

git rev-parse HEAD > reports/runpod/git_commit.txt
python -m pip freeze > reports/runpod/pip_freeze.txt
nvidia-smi > reports/runpod/nvidia_smi.txt || true

echo "RunPod environment setup complete"
