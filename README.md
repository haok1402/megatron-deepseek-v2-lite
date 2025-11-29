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
 [2025-11-28 18:48:17] iteration       50/    5000 | consumed samples:        51200 | elapsed time per iteration (ms): 61674.3 | throughput per GPU (TFLOP/s/GPU): 68.2
 [2025-11-28 18:53:25] iteration       55/    5000 | consumed samples:        56320 | elapsed time per iteration (ms): 61614.9 | throughput per GPU (TFLOP/s/GPU): 68.2
 [2025-11-28 18:58:34] iteration       60/    5000 | consumed samples:        61440 | elapsed time per iteration (ms): 61681.5 | throughput per GPU (TFLOP/s/GPU): 68.2
```
