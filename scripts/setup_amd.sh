#!/usr/bin/env bash
set -euo pipefail

python -m pip install --no-cache-dir \
  "trl==0.24.0" \
  "sentencepiece>=0.2,<1" \
  "scikit-learn>=1.5,<2"

# The AMD base image ships blinker 1.4 through distutils, which pip cannot
# uninstall. Overlay a current wheel first, then install Streamlit normally.
python -m pip install --no-cache-dir --ignore-installed "blinker>=1.9,<2"
python -m pip install --no-cache-dir "streamlit>=1.40,<2"

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
