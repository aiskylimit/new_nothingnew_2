#!/usr/bin/env bash
# CKA diagnostic, then matched uniform/fixed non-uniform GPT-2 DistiLLM and FDD runs.
set -euo pipefail
set -m

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
BASE_PORT=${BASE_PORT:-20120}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Invalid RUN_ID" >&2
    exit 2
fi
RUN_ROOT="${BASE_PATH}/results/gpt2/train/pairing_runs/${RUN_ID}"
LOG_DIR="${BASE_PATH}/logs/gpt2_pairing_ablation/${RUN_ID}"
mkdir -p "$(dirname "${RUN_ROOT}")"
if ! mkdir "${RUN_ROOT}"; then
    echo "Run output already exists: ${RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
export PET_STANDALONE=1
export PYTHONPATH="${BASE_PATH}"

echo "Measuring layer CKA and writing fixed non-uniform pairing on GPU 6; run ${RUN_ID}. Logs: ${LOG_DIR}"
CUDA_VISIBLE_DEVICES=6 python "${BASE_PATH}/tools/cka_layer_pairing.py" \
    --teacher "${TEACHER_PATH}" \
    --student "${STUDENT_PATH}" \
    --data "${RAW_DATA_ROOT}/dolly/raw.jsonl" \
    --output "${RUN_ROOT}/cka" \
    > "${LOG_DIR}/cka.log" 2>&1
PAIRING_FILE="${RUN_ROOT}/cka/nonuniform_pairing.json"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill -- "-${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

pretrain() {
    local pairing=$1 port=$2
    local options=()
    if [ "${pairing}" = nonuniform ]; then
        options+=(--layer-pairing-file "${PAIRING_FILE}")
    fi
    bash "${BASE_PATH}/scripts/gpt2/train_velocity_field.sh" "${BASE_PATH}" "${port}" 1 \
        --model-path "${STUDENT_PATH}" --teacher-model-path "${TEACHER_PATH}" \
        --representation-model velocity --velocity-train-time uniform \
        --save "${RUN_ROOT}/${pairing}/velocity_field" "${options[@]}"
}

CUDA_VISIBLE_DEVICES=6 pretrain uniform "$((BASE_PORT + 0))" > "${LOG_DIR}/pretrain_uniform.log" 2>&1 &
uniform_pid=$!
PIDS+=("${uniform_pid}")
CUDA_VISIBLE_DEVICES=7 pretrain nonuniform "$((BASE_PORT + 1))" > "${LOG_DIR}/pretrain_nonuniform.log" 2>&1 &
nonuniform_pid=$!
PIDS+=("${nonuniform_pid}")
wait "${uniform_pid}"
wait "${nonuniform_pid}"
PIDS=()

train_and_eval() {
    local method=$1 pairing=$2 port=$3 eval_port=$4
    local script ckpt
    if [ "${method}" = distillm ]; then
        script="${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh"
        ckpt="pairing_runs/${RUN_ID}/${pairing}/distillm/0.1B_1.5B"
    elif [ "${method}" = fdd ]; then
        script="${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh"
        ckpt="pairing_runs/${RUN_ID}/${pairing}/fdd/base"
    else
        echo "Unknown method: ${method}" >&2
        return 2
    fi
    local options=()
    if [ "${pairing}" = nonuniform ]; then
        options+=(--layer-pairing-file "${PAIRING_FILE}")
    fi
    bash "${script}" "${BASE_PATH}" "${port}" 1 \
        --model-path "${STUDENT_PATH}" --teacher-model-path "${TEACHER_PATH}" \
        --representation-model velocity --velocity-train-time uniform \
        --contra-curriculum target \
        --velocity-field-path "${RUN_ROOT}/${pairing}/velocity_field/velocity_field.pth" \
        --projector-path "${RUN_ROOT}/${pairing}/velocity_field/projector.pth" \
        --save "${BASE_PATH}/results/gpt2/train/${ckpt}" "${options[@]}"
    for benchmark in dolly self_inst vicuna sinst; do
        for seed in 10 20 30 40 50; do
            bash "${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh" \
                "${BASE_PATH}" "${eval_port}" 1 "${ckpt}" \
                --seed "${seed}" --eval-batch-size 64 \
                --save "${RUN_ROOT}/${pairing}/eval/${method}/${benchmark}/seed_${seed}"
        done
    done
}

for method in distillm fdd; do
    CUDA_VISIBLE_DEVICES=6 train_and_eval "${method}" uniform "$((BASE_PORT + 10))" "$((BASE_PORT + 20))" \
        > "${LOG_DIR}/${method}_uniform.log" 2>&1 &
    uniform_pid=$!
    PIDS+=("${uniform_pid}")
    CUDA_VISIBLE_DEVICES=7 train_and_eval "${method}" nonuniform "$((BASE_PORT + 11))" "$((BASE_PORT + 21))" \
        > "${LOG_DIR}/${method}_nonuniform.log" 2>&1 &
    nonuniform_pid=$!
    PIDS+=("${nonuniform_pid}")
    echo "Training ${method}: uniform GPU 6, fixed non-uniform GPU 7. Outputs: ${RUN_ROOT}"
    wait "${uniform_pid}"
    wait "${nonuniform_pid}"
    PIDS=()
done
echo "Pairing ablation complete."
