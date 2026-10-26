#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON_BIN:-python3.10}
"$PY" -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip wheel setuptools packaging ninja pybind11
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1
# The lock is the actual validated Linux/Python-3.10/CUDA environment.
# Install compiled Transformer Engine against your local CUDA/CUDNN first.
"$PY" -m pip install -r runtime/tanhe/requirements_resolved.txt --extra-index-url https://download.pytorch.org/whl/cu126
"$PY" -m pip check
