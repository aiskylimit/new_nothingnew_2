#!/usr/bin/env bash
# CKA-selected pairing: field pretraining, DistiLLM, FDD, evaluation on GPU 6.
set -euo pipefail

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
BASE_PORT=${BASE_PORT:-20120}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Invalid RUN_ID" >&2
    exit 2
fi
RUN_ROOT="${BASE_PATH}/results/gpt2/train/cka_pairing_runs/${RUN_ID}"
LOG_DIR="${BASE_PATH}/logs/gpt2_cka_pairing/${RUN_ID}"
mkdir -p "$(dirname "${RUN_ROOT}")"
if ! mkdir "${RUN_ROOT}"; then
    echo "Run output already exists: ${RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="${BASE_PATH}"
export PET_STANDALONE=1
echo "CKA pairing end-to-end on GPU 6; run ${RUN_ID}. Logs: ${LOG_DIR}"

python "${BASE_PATH}/tools/cka_layer_pairing.py" \
    --teacher "${TEACHER_PATH}" \
    --student "${STUDENT_PATH}" \
    --data "${RAW_DATA_ROOT}/dolly/raw.jsonl" \
    --output "${RUN_ROOT}/cka" \
    > "${LOG_DIR}/cka.log" 2>&1
PAIRING_FILE="${RUN_ROOT}/cka/cka_pairing.json"
FIELD_PATH="${RUN_ROOT}/velocity_field"

bash "${BASE_PATH}/scripts/gpt2/train_velocity_field.sh" "${BASE_PATH}" "${BASE_PORT}" 1 \
    --model-path "${STUDENT_PATH}" --teacher-model-path "${TEACHER_PATH}" \
    --representation-model velocity --velocity-train-time uniform \
    --layer-pairing-file "${PAIRING_FILE}" \
    --save "${FIELD_PATH}" \
    > "${LOG_DIR}/pretrain.log" 2>&1

for method in distillm fdd; do
    if [ "${method}" = distillm ]; then
        script="${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh"
        ckpt="cka_pairing_runs/${RUN_ID}/distillm/0.1B_1.5B"
    else
        script="${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh"
        ckpt="cka_pairing_runs/${RUN_ID}/fdd/base"
    fi
    bash "${script}" "${BASE_PATH}" "$((BASE_PORT + 10))" 1 \
        --model-path "${STUDENT_PATH}" --teacher-model-path "${TEACHER_PATH}" \
        --representation-model velocity --velocity-train-time uniform \
        --contra-curriculum target \
        --layer-pairing-file "${PAIRING_FILE}" \
        --velocity-field-path "${FIELD_PATH}/velocity_field.pth" \
        --projector-path "${FIELD_PATH}/projector.pth" \
        --save "${BASE_PATH}/results/gpt2/train/${ckpt}" \
        > "${LOG_DIR}/${method}.log" 2>&1
    for benchmark in dolly self_inst vicuna sinst; do
        for seed in 10 20 30 40 50; do
            bash "${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh" \
                "${BASE_PATH}" "$((BASE_PORT + 20))" 1 "${ckpt}" \
                --seed "${seed}" --eval-batch-size 64 \
                --save "${RUN_ROOT}/eval/${method}/${benchmark}/seed_${seed}" \
                >> "${LOG_DIR}/${method}.log" 2>&1
        done
    done
    echo "Completed ${method} on GPU 6."
done
echo "CKA pairing run complete. Results: ${RUN_ROOT}"
