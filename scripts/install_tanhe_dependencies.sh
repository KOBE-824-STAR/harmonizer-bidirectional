#!/usr/bin/env bash
set -euo pipefail
cd /data/juziwei/hlk/double_DiT
PY=runtime/tanhe/venv/bin/python
# Provision Python 3.10 venv and CUDA 12.8 first.
# CUDA wheels are explicit; requirements_resolved.txt records the verified environment.
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1+cu126' 'torchvision==0.22.1+cu126'
"$PY" -m pip install 'numpy==1.26.4' 'attrs==25.1.0' 'diffusers==0.33.1' 'transformers==4.51.3' 'huggingface-hub==0.32.4' 'peft==0.14.0' 'einops==0.8.1' 'omegaconf==2.3.0' 'hydra-core==1.3.2' 'loguru==0.7.2' 'termcolor==2.5.0' 'iopath==0.1.10' 'fvcore==0.1.5.post20221221' 'safetensors==0.5.3' 'sentencepiece==0.2.0' 'lpips==0.1.4' 'pillow==11.1.0' 'ftfy==6.3.1' 'imageio==2.37.0' 'mediapy==1.2.2' 'nvidia-ml-py==12.535.133' 'decord==0.6.0' 'webdataset==0.2.111' 'boto3==1.35.99' 'av' 'qwen-vl-utils==0.0.11' 'ninja' 'wheel' 'pybind11' 'packaging'
"$PY" -m pip install --no-deps megatron-core==0.10.0
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PWD/runtime/tanhe/venv/bin:$PATH"
export NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=89 MAX_JOBS=4 NVTE_BUILD_THREADS_PER_JOB=4
export CUDNN_PATH="$PWD/runtime/tanhe/venv/lib/python3.10/site-packages/nvidia/cudnn"
"$PY" -m pip install --no-build-isolation 'transformer-engine[pytorch]==2.1.0'
"$PY" -m pip install 'pandas==2.2.3' 'opencv-python==4.10.0.84' 'imageio-ffmpeg==0.6.0'
"$PY" -m pip freeze > runtime/tanhe/validated_constraints.txt
"$PY" -m pip install -c runtime/tanhe/validated_constraints.txt 'nvidia-modelopt[torch]==0.19.0' flask-restful nltk pytest pytest-cov pytest-mock pytest-random-order 'tensorstore==0.1.45' tiktoken wandb wrapt 'zarr<3'
"$PY" -m pip check
"$PY" -m pip freeze > runtime/tanhe/requirements_resolved.txt
