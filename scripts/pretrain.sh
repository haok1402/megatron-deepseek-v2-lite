#!/bin/bash
# Pretrain with DeepSeek-V2-Lite.

source scripts/activate.sh

declare -A MODEL_CONFIG
declare -A TRAIN_CONFIG
declare -A INFRA_CONFIG

# Infrastructure.
INFRA_CONFIG[bf16]=true
INFRA_CONFIG[moe-router-dtype]=fp32
INFRA_CONFIG[moe-grouped-gemm]=true
INFRA_CONFIG[moe-router-fusion]=true
INFRA_CONFIG[moe-permute-fusion]=true
INFRA_CONFIG[moe-shared-expert-overlap]=true
INFRA_CONFIG[moe-token-dispatcher-type]=alltoall
INFRA_CONFIG[use-distributed-optimizer]=true
INFRA_CONFIG[overlap-param-gather]=true
INFRA_CONFIG[overlap-grad-reduce]=true
INFRA_CONFIG[cross-entropy-loss-fusion]=true
INFRA_CONFIG[cross-entropy-fusion-impl]=te
INFRA_CONFIG[expert-model-parallel-size]=2
INFRA_CONFIG[pipeline-model-parallel-size]=2

# Embedding.
MODEL_CONFIG[tokenizer-type]=HuggingFaceTokenizer
MODEL_CONFIG[tokenizer-model]=deepseek-ai/DeepSeek-V2-Lite
MODEL_CONFIG[vocab-size]=102400
MODEL_CONFIG[position-embedding-type]=rope
MODEL_CONFIG[rotary-base]=10000
MODEL_CONFIG[max-position-embeddings]=163840

# Multi-Latent Attention.
MODEL_CONFIG[enable-experimental]=true
MODEL_CONFIG[multi-latent-attention]=true
MODEL_CONFIG[qk-head-dim]=128
MODEL_CONFIG[qk-pos-emb-head-dim]=64
MODEL_CONFIG[kv-lora-rank]=512
MODEL_CONFIG[num-attention-heads]=16
MODEL_CONFIG[v-head-dim]=128
MODEL_CONFIG[qk-layernorm]=true
MODEL_CONFIG[normalization]=RMSNorm
MODEL_CONFIG[norm-epsilon]=1e-6
MODEL_CONFIG[attention-dropout]=0.0

# Feedforward Network.
MODEL_CONFIG[num-layers]=28
MODEL_CONFIG[disable-bias-linear]=true
MODEL_CONFIG[hidden-size]=2048
MODEL_CONFIG[ffn-hidden-size]=10944
MODEL_CONFIG[hidden-dropout]=0.0
MODEL_CONFIG[swiglu]=true
MODEL_CONFIG[moe-layer-freq]="([0]+[1]*27)"
MODEL_CONFIG[num-experts]=64
MODEL_CONFIG[moe-ffn-hidden-size]=1408
MODEL_CONFIG[moe-shared-expert-intermediate-size]=$((1408 * 2))
MODEL_CONFIG[moe-router-dtype]=fp32
MODEL_CONFIG[moe-router-score-function]=softmax
MODEL_CONFIG[moe-router-topk]=6

# Learning.
TRAIN_CONFIG[train-iters]=5000
TRAIN_CONFIG[micro-batch-size]=1
TRAIN_CONFIG[global-batch-size]=1024
TRAIN_CONFIG[seq-length]=2048
TRAIN_CONFIG[lr]=3e-4
TRAIN_CONFIG[min-lr]=3e-5
TRAIN_CONFIG[lr-warmup-iters]=200
TRAIN_CONFIG[lr-decay-iters]=4800
TRAIN_CONFIG[lr-decay-style]=cosine
TRAIN_CONFIG[init-method-std]=0.02
TRAIN_CONFIG[optimizer]=adam
TRAIN_CONFIG[log-interval]=5
TRAIN_CONFIG[log-throughput]=true

# Dataset.
DATA_ARGS_PATH=$(mktemp)
find $WORKSPACE/dataset/toktxt/ -type f -name "*.idx" | sort | while read -r FILE; do
    printf "1.0 %s " ${FILE%.idx} >> $DATA_ARGS_PATH
done
TRAIN_CONFIG[data-args-path]=$DATA_ARGS_PATH
TRAIN_CONFIG[split]=969,30,1

# Assemble the arguments.
MAIN_ARGS=()
for key in ${!MODEL_CONFIG[@]}; do
    val=${MODEL_CONFIG[$key]}
    [[ $val == true ]] && MAIN_ARGS+=(--$key) || MAIN_ARGS+=(--$key $val)
done
for key in ${!TRAIN_CONFIG[@]}; do
    val=${TRAIN_CONFIG[$key]}
    [[ $val == true ]] && MAIN_ARGS+=(--$key) || MAIN_ARGS+=(--$key $val)
done
for key in ${!INFRA_CONFIG[@]}; do
    val=${INFRA_CONFIG[$key]}
    [[ $val == true ]] && MAIN_ARGS+=(--$key) || MAIN_ARGS+=(--$key $val)
done

# Launch the training.
TRUN_ARGS=()
TRUN_ARGS+=(--nnodes 1 --node-rank 0 --nproc-per-node 8)
TRUN_ARGS+=(--rdzv-backend c10d --rdzv-endpoint localhost:29500)
torchrun ${TRUN_ARGS[@]} -m pretrain_gpt ${MAIN_ARGS[@]} 2>&1 | tee $WORKSPACE/pretrain.log
