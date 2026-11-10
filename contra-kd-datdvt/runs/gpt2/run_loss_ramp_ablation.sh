#!/usr/bin/env bash
# Pretrain the submission-style velocity field, then run loss-ramp methods in parallel.
set -euo pipefail
set -m

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
BASE_PORT=${BASE_PORT:-20120}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "RUN_ID may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi
RUN_ROOT="${BASE_PATH}/results/gpt2/train/loss_ramp_runs/${RUN_ID}"
LOG_DIR="${LOG_DIR:-${BASE_PATH}/logs/gpt2_loss_ramp_ablation}/${RUN_ID}"
VELOCITY_PATH="${RUN_ROOT}/velocity_field_uniform"
mkdir -p "$(dirname "${RUN_ROOT}")"
if ! mkdir "${RUN_ROOT}"; then
    echo "Run output already exists; choose a new RUN_ID: ${RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
export PET_STANDALONE=1
export PYTHONPATH="${BASE_PATH}"

echo "Pretraining submission-style velocity field with uniform t on GPU 6..."
CUDA_VISIBLE_DEVICES=6 bash "${BASE_PATH}/scripts/gpt2/train_velocity_field.sh" \
    "${BASE_PATH}" "${BASE_PORT}" 1 \
    --model-path "${STUDENT_PATH}" \
    --teacher-model-path "${TEACHER_PATH}" \
    --representation-model velocity \
    --velocity-train-time uniform \
    --save "${VELOCITY_PATH}" \
    > "${LOG_DIR}/pretrain.log" 2>&1

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill -- "-${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_method() {
    local method=$1
    local master_port=$2
    local eval_port=$3
    local script ckpt
    if [ "${method}" = distillm ]; then
        script="${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh"
        ckpt="loss_ramp_runs/${RUN_ID}/distillm/0.1B_1.5B"
    else
        script="${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh"
        ckpt="loss_ramp_runs/${RUN_ID}/fdd/base"
    fi

    bash "${script}" "${BASE_PATH}" "${master_port}" 1 \
        --model-path "${STUDENT_PATH}" \
        --teacher-model-path "${TEACHER_PATH}" \
        --representation-model velocity \
        --velocity-train-time uniform \
        --contra-curriculum loss-ramp \
        --velocity-field-path "${VELOCITY_PATH}/velocity_field.pth" \
        --projector-path "${VELOCITY_PATH}/projector.pth" \
        --save "${BASE_PATH}/results/gpt2/train/${ckpt}"

    for benchmark in dolly self_inst vicuna sinst; do
        for seed in 10 20 30 40 50; do
            bash "${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh" \
                "${BASE_PATH}" "${eval_port}" 1 "${ckpt}" \
                --seed "${seed}" --eval-batch-size 64 \
                --save "${RUN_ROOT}/eval/${method}/${benchmark}/seed_${seed}"
        done
    done
}

CUDA_VISIBLE_DEVICES=6 run_method distillm "$((BASE_PORT + 10))" "$((BASE_PORT + 20))" \
    > "${LOG_DIR}/distillm.log" 2>&1 &
distillm_pid=$!
PIDS+=("${distillm_pid}")

CUDA_VISIBLE_DEVICES=7 run_method fdd "$((BASE_PORT + 11))" "$((BASE_PORT + 21))" \
    > "${LOG_DIR}/fdd.log" 2>&1 &
fdd_pid=$!
PIDS+=("${fdd_pid}")

echo "Run ${RUN_ID}: DistillM GPU 6, FDD GPU 7. Logs: ${LOG_DIR}; outputs: ${RUN_ROOT}"
wait "${distillm_pid}"
wait "${fdd_pid}"
PIDS=()
echo "Both loss-ramp branches completed."
