#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothingnew_2}"
BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/learn2follow}"
export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/mnt/local/uvenvs/learn2follow}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${BASE_DIR}/.cache/uv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NUM_PROCESSES=2
export PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"

export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/models/Qwen3-8B}"
export DATASET_JSONL="${DATASET_JSONL:-${BASE_DIR}/data/hypergraph_openthoughts30k_operations_v3.jsonl}"
export RUN_ROOT="${RUN_ROOT:-${BASE_DIR}/outputs}"
export CONFIG="${CONFIG:-adaptive_graph_ftrl_redesign/configs/hard_experiment.yaml}"
export MAX_STEPS="${MAX_STEPS:-100}"
export SAVE_EVERY="${SAVE_EVERY:-25}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export EVAL_SEED="${EVAL_SEED:-42}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime25,aime26,hmmt25}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"

if [[ ! -f "${DATASET_JSONL}" ]]; then
    echo "Missing operation-annotated training JSONL: ${DATASET_JSONL}. See download.txt." >&2
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model: ${MODEL_PATH}. See download.txt or set MODEL_PATH." >&2
    exit 1
fi

uv sync --frozen --extra cuda --group dev
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh \
    --model="${MODEL_PATH}" --gradient-accumulation-steps=16 "$@"

# Results are written by the retained evaluator under:
# ${PROJECT_ROOT}/full_masked_graph_opsd/outputs/evaluation/<run-name>/
