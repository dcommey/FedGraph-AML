#!/usr/bin/env bash
# Create .venv with the training environment (Python 3.11, CUDA 12.1).
# Usage: bash scripts/setup_environment.sh /path/to/python3.11
set -euo pipefail
base_python=${1:-python3.11}
"$base_python" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__, torch.cuda.is_available())'
