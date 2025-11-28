#!/bin/bash
# Install runtime dependencies.

set -eu

# Create the conda environment.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n megatron python=3.12 -y
conda activate megatron

# We're using PyTorch 2.8.0 with CUDA 12.8.
export CUDA_HOME=/usr/local/cuda-12.8
pip3 install numpy torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# To speed up the building of FlashAttention, ninja must be installed first.
pip3 install ninja packaging psutil
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE"
export FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE"
pip3 install flash-attn==2.8.1 --no-build-isolation

# Pin the version of TransformerEngine to match the PyTorch version.
SITE_PACKAGES=$CONDA_PREFIX/lib/python3.12/site-packages
export CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/nvtx/include:$SITE_PACKAGES/nvidia/cudnn/include:$SITE_PACKAGES/nvidia/nccl/include
export C_INCLUDE_PATH=$SITE_PACKAGES/nvidia/nvtx/include:$SITE_PACKAGES/nvidia/cudnn/include:$SITE_PACKAGES/nvidia/nccl/include
export CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn
pip3 install --no-build-isolation transformer-engine[pytorch]==2.8.0

# Install NVIDIA Apex from source.
git clone https://github.com/NVIDIA/apex
pushd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
popd
rm -rvf apex

# Other dependencies.
pip3 install zstandard six regex pyyaml transformers wandb pybind11 tensorboard

# Apply the megatron patch.
pushd megatron
git apply ../scripts/megatron.patch
popd
