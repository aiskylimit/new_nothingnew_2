#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
source /mnt/local/uvenvs/learn2follow/bin/activate

PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothingnew_2}"
BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/learn2follow}"
export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHON_BIN="$(command -v python)"

export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/models/Qwen3-8B}"
export RAW_DATA_ROOT="${RAW_DATA_ROOT:-${BASE_DIR}/data/raw}"
export PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-${BASE_DIR}/data/processed}"
export EVAL_DATASET_DIR="${EVAL_DATASET_DIR:-${PREPARED_DATA_ROOT}/eval}"
RUN_DIR="${RUN_DIR:-${BASE_DIR}/outputs/graph_hard_rho0.9_rms0.9_floor0.0001_clip1_duel16_pool128_min8_w0_H5_aw3_clr0.01_tau0.05_lgt0.01_gam0.95_b32_lr5e-06_ans1_s42_es42_n100_916885eda94a_20261011_064730_236603}"
export EVAL_SEED="${EVAL_SEED:-42}"
export EVAL_STEPS="${EVAL_STEPS:-25,50,75,100}"
export EVAL_DATASETS="${EVAL_DATASETS:-aime25,aime26,hmmt25}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing local model: ${MODEL_PATH}. See download.txt or set MODEL_PATH." >&2
    exit 1
fi
IFS=',' read -r -a checkpoints <<< "${EVAL_STEPS}"
for step in "${checkpoints[@]}"; do
    if [[ ! -f "${RUN_DIR}/checkpoint-${step}/adapter_model.safetensors" ]]; then
        echo "Missing checkpoint adapter: ${RUN_DIR}/checkpoint-${step}/adapter_model.safetensors" >&2
        exit 1
    fi
done

# Short socket paths avoid ZeroMQ's 107-byte Unix IPC limit.
export TMPDIR="$(mktemp -d /tmp/l2f-eval-XXXXXX)"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"

# Prepare benchmark data entirely from local snapshots, as in OPSD.
IFS=',' read -r -a datasets <<< "${EVAL_DATASETS}"
for dataset in "${datasets[@]}"; do
    if [[ ! -d "${EVAL_DATASET_DIR}/${dataset}" ]]; then
        "${PYTHON_BIN}" data/prepare_data.py \
            --raw_root "${RAW_DATA_ROOT}" --output_root "${PREPARED_DATA_ROOT}" \
            --only_eval "${dataset}"
    fi
done

# Evaluate the saved adapters directly; no training is launched.
"${PYTHON_BIN}" -m adaptive_graph_ftrl_redesign.evaluate \
    --run-dir "${RUN_DIR}" \
    --steps "${EVAL_STEPS}" \
    --datasets "${EVAL_DATASETS}" \
    --dataset-dir "${EVAL_DATASET_DIR}" \
    --model "${MODEL_PATH}" \
    --gpus "${CUDA_VISIBLE_DEVICES}" \
    --seed "${EVAL_SEED}" \
    --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
    --launcher python "$@"

# Results are written by the retained evaluator under:
# ${PROJECT_ROOT}/full_masked_graph_opsd/outputs/evaluation/<run-name>/
