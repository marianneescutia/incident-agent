#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir -r requirements-amd.txt

python - <<'PY'
import torch
import transformers
import accelerate
import datasets
import peft
import trl
import sklearn

assert torch.cuda.is_available(), "ROCm GPU is not available to PyTorch."
assert torch.version.hip, "This PyTorch build does not report a ROCm/HIP version."

print("PyTorch:", torch.__version__)
print("ROCm/HIP:", torch.version.hip)
print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)
print("Accelerate:", accelerate.__version__)
print("Datasets:", datasets.__version__)
print("TRL:", trl.__version__)
print("scikit-learn:", sklearn.__version__)
print("GPU memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY
