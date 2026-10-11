#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:--1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export VAL_N="${VAL_N:-12}"

exec bash graph_based_scripts/run_graph_based_eval.sh "$@"
