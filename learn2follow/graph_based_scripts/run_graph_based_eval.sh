#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-dsd}" uv run python eval/evaluate_math.py \
    --base_model "${MODEL_PATH:-Qwen/Qwen3-4B}" \
    --checkpoint_dir "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a graph-based checkpoint}" \
    --dataset "${DATASET:-aime24}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-38912}" \
    --temperature "${TEMPERATURE:-0.6}" \
    --top_p "${TOP_P:-0.95}" \
    --top_k "${TOP_K:-20}" \
    --min_p "${MIN_P:-0}" \
    --presence_penalty "${PRESENCE_PENALTY:-0}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-4}" \
    --val_n "${VAL_N:-8}" \
    --seed "${EVAL_SEED:-42}" \
    "$@"
