#!/bin/bash
# Pretrain the DeepSeek-V2-Lite model.

source scripts/activate.sh

declare -A MODEL_CONFIG
declare -A TRAIN_CONFIG
declare -A INFRA_CONFIG

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
INFRA_CONFIG[num-virtual-stages-per-pipeline-rank]=2

MODEL_CONFIG[tokenizer-type]=HuggingFaceTokenizer
MODEL_CONFIG[tokenizer-model]=deepseek-ai/DeepSeek-V2-Lite
MODEL_CONFIG[vocab-size]=102400
MODEL_CONFIG[position-embedding-type]=rope
MODEL_CONFIG[rotary-base]=10000
MODEL_CONFIG[max-position-embeddings]=163840

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

DATA_ARGS_PATH=$(mktemp)
find $WORKSPACE/dataset/toktxt/ -type f -name "*.idx" | sort | while read -r FILE; do
    printf "1.0 %s " ${FILE%.idx} >> $DATA_ARGS_PATH
done
TRAIN_CONFIG[data-args-path]=$DATA_ARGS_PATH
TRAIN_CONFIG[split]=969,30,1

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

SLURM_NNODES=${SLURM_NNODES:-1}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}}
SLURM_STEP_NODELIST=${SLURM_STEP_NODELIST:-$(hostname)}

NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
NPROC_PER_NODE=$(echo "$SLURM_STEP_GPUS" | tr ',' '\n' | wc -l)
RDZV_BACKEND=c10d
RDZV_ENDPOINT=$(command -v scontrol &>/dev/null && scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1 || echo localhost):15213

TRUN_ARGS=()
TRUN_ARGS+=(--nnodes=$NNODES --node-rank=$NODE_RANK --nproc-per-node=$NPROC_PER_NODE)
TRUN_ARGS+=(--rdzv-backend=$RDZV_BACKEND --rdzv-endpoint=$RDZV_ENDPOINT)

SCRIPT=megatron/pretrain_gpt.py
OUTPUT=$WORKSPACE/pretrain_${NODE_RANK}.log

torchrun ${TRUN_ARGS[@]} $SCRIPT ${MAIN_ARGS[@]} 2>&1 | tee $OUTPUT
