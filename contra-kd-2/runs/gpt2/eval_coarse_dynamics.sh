#!/usr/bin/env bash
set -euo pipefail
set -m

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"

PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothingnew_2}"
BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/contra-kd-2}"
ENV_DIR="${ENV_DIR:-/mnt/local/uvenvs/contra}"
if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
    echo "Missing environment: ${ENV_DIR}. Provision contra.txt before running offline." >&2
    exit 1
fi
source "${ENV_DIR}/bin/activate"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

TEACHER_PATH="${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}"
TRAIN_ROOT="${FDD_STUDY_TRAIN_ROOT:-${BASE_PATH}/results/gpt2/train/fdd_seed_study}"
DATA_PATH="${COARSE_DYNAMICS_DATA_PATH:-${BASE_DIR}/data/raw/dolly/valid.jsonl}"
SAVE_PATH="${COARSE_DYNAMICS_OUTPUT:-${BASE_PATH}/results/gpt2/eval_main/fdd_seed_study/coarse_dynamics}"
read -r -a SEEDS <<< "${COARSE_DYNAMICS_SEEDS:-1024 2048 4096}"
read -r -a GPUS <<< "${COARSE_DYNAMICS_GPUS:-4 5}"
if [[ ${#GPUS[@]} -ne 2 || ! ${GPUS[0]} =~ ^[0-9]+$ || ! ${GPUS[1]} =~ ^[0-9]+$ || ${GPUS[0]} == "${GPUS[1]}" ]]; then
    echo "COARSE_DYNAMICS_GPUS must list two distinct physical GPU indices." >&2
    exit 2
fi

required=("${TEACHER_PATH}/config.json" "${DATA_PATH}")
for seed in "${SEEDS[@]}"; do
    required+=("${TRAIN_ROOT}/fdd/${seed}/config.json" "${TRAIN_ROOT}/contra_fdd/${seed}/config.json")
done
for file in "${required[@]}"; do
    if [[ ! -s "${file}" ]]; then
        echo "Missing coarse-dynamics input: ${file}" >&2
        exit 1
    fi
done

mkdir -p "${SAVE_PATH}/logs"
pids=()
cleanup() {
    local i pid
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        kill -TERM -- "-${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_gpu() {
    local worker=$1 i seed log
    export CUDA_VISIBLE_DEVICES="${GPUS[$worker]}"
    for ((i = worker; i < ${#SEEDS[@]}; i += ${#GPUS[@]})); do
        seed=${SEEDS[$i]}
        log="${SAVE_PATH}/logs/${seed}.log"
        echo "Evaluating FDD and FDD + Contra seed ${seed} on GPU ${CUDA_VISIBLE_DEVICES}. See ${log}"
        if ! python3 -u "${BASE_PATH}/evaluate_coarse_dynamics.py" \
            --teacher-model-path "${TEACHER_PATH}" --baseline-model-path "${TRAIN_ROOT}/fdd/${seed}" \
            --model-path "${TRAIN_ROOT}/contra_fdd/${seed}" --data-path "${DATA_PATH}" \
            --save "${SAVE_PATH}/${seed}" --seed "${seed}" \
            --num-calibration "${COARSE_DYNAMICS_CALIBRATION:-128}" --num-eval "${COARSE_DYNAMICS_EVAL:-256}" \
            --eval-batch-size "${COARSE_DYNAMICS_BATCH_SIZE:-8}" > "${log}" 2>&1; then
            echo "Coarse-dynamics evaluation failed for seed ${seed}. See ${log}" >&2
            return 1
        fi
        echo "Completed seed ${seed}."
    done
}

for worker in "${!GPUS[@]}"; do
    if ((worker < ${#SEEDS[@]})); then
        run_gpu "${worker}" &
        pids+=("$!")
    fi
done
while ((${#pids[@]})); do
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        if ! kill -0 "${pid}" 2>/dev/null; then
            unset 'pids[i]'
            wait "${pid}"
        fi
    done
    if ((${#pids[@]})); then
        sleep 1
    fi
done
