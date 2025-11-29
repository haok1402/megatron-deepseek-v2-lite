# megatron-deepseek-v2-lite

A minimal codebase for pretraining DeepSeek-V2-Lite with Megatron-LM.

## Getting Started

First, set up the Conda environment with the installation script.

```
bash scripts/install.sh
```

Next, preprocess the training corpus.

```
bash scripts/preprocess.sh
```

Finally, launch the pretraining.

```
bash scripts/pretrain.sh
```

## Performance Metrics

(DP2, PP2, EP2) on 8xH100 GPUs.

```
 [2025-11-28 21:47:03] iteration      220/    5000 | elapsed time per iteration (ms): 61272.8 | throughput per GPU (TFLOP/s/GPU): 68.6
 [2025-11-28 21:52:08] iteration      225/    5000 | elapsed time per iteration (ms): 61195.4 | throughput per GPU (TFLOP/s/GPU): 68.7
 [2025-11-28 21:57:14] iteration      230/    5000 | elapsed time per iteration (ms): 61112.0 | throughput per GPU (TFLOP/s/GPU): 68.8
```
