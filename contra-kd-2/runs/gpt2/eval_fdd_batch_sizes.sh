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
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export RAW_DATA_ROOT="${RAW_DATA_ROOT:-${BASE_DIR}/data/raw}"
# Give each single-node torchrun its own rendezvous on a free local port.
export PET_STANDALONE=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

TRAIN_ROOT="${FDD_STUDY_TRAIN_ROOT:-${BASE_PATH}/results/gpt2/train/fdd_seed_study}"
EVAL_ROOT="${BASE_PATH}/results/gpt2/eval_main/fdd_seed_study"
SWEEP_ROOT="${EVAL_ROOT}/batch_size_sweep"
BASE_PORT="${BASE_PORT:-20140}"
TRAIN_SEEDS=(1024 2048 4096)
EVAL_SEEDS=(10 20 30 40 50)
BATCH_SIZES=(32 16 8)
BENCHMARKS=(dolly self_inst vicuna sinst)
DATA_NAMES=(dolly self_inst vicuna sinst_11_)
DATA_DIRS=(dolly self-inst vicuna sinst/11_)

read -r -a GPUS <<< "${FDD_BATCH_SWEEP_GPUS:-4 5 6}"
if (( ${#GPUS[@]} < 1 || ${#GPUS[@]} > ${#TRAIN_SEEDS[@]} )); then
    echo "FDD_BATCH_SWEEP_GPUS must list one to three distinct physical GPU indices." >&2
    exit 2
fi
seen=" "
for gpu in "${GPUS[@]}"; do
    if [[ ! ${gpu} =~ ^(0|[1-9][0-9]*)$ || ${seen} == *" ${gpu} "* ]]; then
        echo "FDD_BATCH_SWEEP_GPUS must list one to three distinct physical GPU indices." >&2
        exit 2
    fi
    seen+="${gpu} "
done

required=()
for seed in "${TRAIN_SEEDS[@]}"; do
    required+=("${TRAIN_ROOT}/contra_fdd/${seed}/config.json")
done
for data_dir in "${DATA_DIRS[@]}"; do
    required+=("${RAW_DATA_ROOT}/${data_dir}/valid.jsonl")
done
for file in "${required[@]}"; do
    if [[ ! -s "${file}" ]]; then
        echo "Missing batch-size sweep input: ${file}" >&2
        exit 1
    fi
done

mkdir -p "${SWEEP_ROOT}/logs"
LOCK_DIR="${SWEEP_ROOT}/.running"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Batch-size sweep is locked at ${LOCK_DIR}. Check whether its launcher is still running." >&2
    exit 1
fi
pids=()
cleanup() {
    local i
    for i in "${!pids[@]}"; do
        kill -TERM -- "-${pids[$i]}" 2>/dev/null || true
        wait "${pids[$i]}" 2>/dev/null || true
    done
    rmdir "${LOCK_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_seed() {
    local seed=$1 port=$2 batch_size i eval_seed save_path result
    local ckpt_name="contra_fdd/${seed}"
    for batch_size in "${BATCH_SIZES[@]}"; do
        save_path="${SWEEP_ROOT}/bs${batch_size}/generation"
        for i in "${!BENCHMARKS[@]}"; do
            for eval_seed in "${EVAL_SEEDS[@]}"; do
                result="${save_path}/${ckpt_name}/${DATA_NAMES[$i]}-512/${eval_seed}"
                if [[ -s "${result}/metrics.json" && -s "${result}/args.json" ]]; then
                    continue
                fi
                echo "Evaluating seed ${seed}, batch size ${batch_size}, ${DATA_NAMES[$i]}, generation seed ${eval_seed}."
                bash "scripts/gpt2/eval/eval_main_${BENCHMARKS[$i]}.sh" "${BASE_PATH}" "${port}" 1 \
                    "fdd_seed_study/${ckpt_name}" --model-path "${TRAIN_ROOT}/${ckpt_name}" \
                    --ckpt-name "${ckpt_name}" --save "${save_path}" \
                    --seed "${eval_seed}" --eval-batch-size "${batch_size}" || return
                if [[ ! -s "${result}/metrics.json" || ! -s "${result}/args.json" ]]; then
                    echo "Missing evaluation output in ${result}." >&2
                    return 1
                fi
            done
        done
    done
}

run_gpu() {
    local worker=$1 i seed log
    export CUDA_VISIBLE_DEVICES="${GPUS[$worker]}"
    for ((i = worker; i < ${#TRAIN_SEEDS[@]}; i += ${#GPUS[@]})); do
        seed=${TRAIN_SEEDS[$i]}
        log="${SWEEP_ROOT}/logs/contra_fdd_${seed}.log"
        echo "Evaluating FDD + CONTRA seed ${seed} on GPU ${CUDA_VISIBLE_DEVICES}. See ${log}"
        if ! run_seed "${seed}" "$((BASE_PORT + worker))" >> "${log}" 2>&1; then
            echo "Batch-size sweep failed for seed ${seed}. See ${log}" >&2
            return 1
        fi
        echo "Completed seed ${seed}."
    done
}

for worker in "${!GPUS[@]}"; do
    run_gpu "${worker}" &
    pids+=("$!")
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

echo "Batch-size sweep complete. Results: ${SWEEP_ROOT}"
