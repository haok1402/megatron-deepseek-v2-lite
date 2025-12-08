#!/bin/bash
# Prepare the dataset for training.

source scripts/activate.sh

# Download one shard of the DCLM dataset from HuggingFace.
FILE=global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst
mkdir -p $WORKSPACE/dataset/rawtxt

MAIN_ARGS=()
MAIN_ARGS+=(--include $FILE)
MAIN_ARGS+=(--local-dir $WORKSPACE/dataset/rawtxt)
MAIN_ARGS+=(--repo-type dataset)
MAIN_ARGS+=(mlfoundations/dclm-baseline-1.0)
hf download ${MAIN_ARGS[@]}

# Decompress that shard for preprocessing with Megatron-LM.
unzstd -f $WORKSPACE/dataset/rawtxt/$FILE
rm $WORKSPACE/dataset/rawtxt/$FILE

# Run the tokenization script for training DeepSeek-V2-Lite.
FILE=${FILE%.zst}
mkdir -p $WORKSPACE/dataset/toktxt/global-shard_01_of_10/local-shard_0_of_10/

MAIN_ARGS=()
MAIN_ARGS+=(--input $WORKSPACE/dataset/rawtxt/$FILE)
MAIN_ARGS+=(--output-prefix $WORKSPACE/dataset/toktxt/${FILE%.jsonl})
MAIN_ARGS+=(--tokenizer-type HuggingFaceTokenizer)
MAIN_ARGS+=(--tokenizer-model deepseek-ai/DeepSeek-V2-Lite)
MAIN_ARGS+=(--append-eod)
MAIN_ARGS+=(--workers 32)
python3 megatron/tools/preprocess_data.py ${MAIN_ARGS[@]}
