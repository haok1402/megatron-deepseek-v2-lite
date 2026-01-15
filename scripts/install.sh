#!/bin/bash
# Install the runtime dependencies for Megatron-LM.

set -eu

conda create -n megatron python=3.12 -y
conda activate megatron

export CUDA_HOME=/usr/local/cuda-12.8
pip3 install numpy torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip3 install ninja packaging psutil
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE"
export FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE"
pip3 install flash-attn==2.8.1 --no-build-isolation

SITE_PACKAGES=$CONDA_PREFIX/lib/python3.12/site-packages
export CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/nvtx/include:$SITE_PACKAGES/nvidia/cudnn/include:$SITE_PACKAGES/nvidia/nccl/include
export C_INCLUDE_PATH=$SITE_PACKAGES/nvidia/nvtx/include:$SITE_PACKAGES/nvidia/cudnn/include:$SITE_PACKAGES/nvidia/nccl/include
export CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn
pip3 install --no-build-isolation transformer-engine[pytorch]==2.8.0

git clone https://github.com/NVIDIA/apex
pushd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
popd
rm -rvf apex

pip3 install zstandard six regex pyyaml transformers wandb pybind11 tensorboard torch==2.8.0

pushd megatron
git apply ../scripts/megatron.patch
popd
