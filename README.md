# Megatron DeepSeek-V2-Lite

Pretrain [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## Requirements

- CUDA 12.8
- Python 3.12

## Usage

Setup environment:

```bash
bash scripts/install.sh
```

Preprocess data:

```bash
bash scripts/preprocess.sh
```

Train:

```bash
bash scripts/pretrain.sh
```

## Configuration

Edit `scripts/pretrain.sh` to modify model architecture, training hyperparameters, and parallelism settings.

Multi-node training is supported via Slurm (`SLURM_NNODES`, `SLURM_NODEID`, `SLURM_STEP_NODELIST`).
