#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
JOB_ID=${JOB_ID:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname "$0")/..:$PYTHONPATH" \
torchrun \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --nproc-per_node "$GPUS" \
    --rdzv_id "$JOB_ID" \
    --rdzv_backend "c10d" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$PORT" \
    "$(dirname "$0")/train.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch \
    "${@:4}"
