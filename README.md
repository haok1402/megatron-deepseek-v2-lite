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
 [2025-11-28 17:17:41] iteration       10/    5000 | consumed samples:        10240 | elapsed time per iteration (ms): 67423.1 | throughput per GPU (TFLOP/s/GPU): 62.4
 [2025-11-28 17:23:34] iteration       15/    5000 | consumed samples:        15360 | elapsed time per iteration (ms): 70593.5 | throughput per GPU (TFLOP/s/GPU): 59.6
```
