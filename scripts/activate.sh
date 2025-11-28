#!/bin/bash
# Activate runtime environment.

set -eu

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate megatron

export WORKSPACE=$PWD/workspace
export CUDA_HOME=/usr/local/cuda-12.8
export PYTHONPATH=$PWD/megatron:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
