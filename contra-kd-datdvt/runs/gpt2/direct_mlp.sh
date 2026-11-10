#!/bin/bash
# Matched-capacity direct teacher-state prediction ablation for GPT-2.
# Usage: bash runs/gpt2/direct_mlp.sh [BASE_PATH] [MASTER_PORT] [pretrain|distillm|fdd]
set -e -o pipefail

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
MASTER_PORT=${2:-2012}
STAGE=${3:-distillm}
DEVICE=${CUDA_VISIBLE_DEVICES:-7}
export CUDA_VISIBLE_DEVICES=${DEVICE}
BASE_DIR=${BASE_DIR:-"/mnt/local/${PROJECT_NAME:-aiskylimit_new_nothingnew_2}/contra-kd"}
export STUDENT_PATH=${STUDENT_PATH:-"${BASE_DIR}/results/gpt2/train/init/gpt2-base"}
export TEACHER_PATH=${TEACHER_PATH:-"${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge"}
export PROCESSED_DATA_ROOT=${PROCESSED_DATA_ROOT:-"${BASE_DIR}/processed_data"}
MLP_PATH="${BASE_PATH}/results/gpt2/train/direct_state_mlp_t0"

for checkpoint in "${STUDENT_PATH}" "${TEACHER_PATH}"; do
    if [ ! -f "${checkpoint}/config.json" ]; then
        echo "Missing checkpoint: ${checkpoint}/config.json" >&2
        exit 1
    fi
done

if [ "${STAGE}" = pretrain ]; then
    bash "${BASE_PATH}/scripts/gpt2/train_velocity_field.sh" "${BASE_PATH}" "${MASTER_PORT}" 1 \
        --model-path "${STUDENT_PATH}" \
        --teacher-model-path "${TEACHER_PATH}" \
        --representation-model direct-mlp \
        --save "${MLP_PATH}"
    exit 0
fi

for checkpoint in "${MLP_PATH}/direct_state_mlp.pth" "${MLP_PATH}/projector.pth"; do
    if [ ! -f "${checkpoint}" ]; then
        echo "Missing pretrained direct MLP checkpoint: ${checkpoint}" >&2
        exit 1
    fi
done

case "${STAGE}" in
    distillm)
        TRAIN_SCRIPT="${BASE_PATH}/scripts/gpt2/distillm/contra_0.1B_1.5B.sh"
        CKPT="direct_mlp_t0/distillm/0.1B_1.5B"
        ;;
    fdd)
        TRAIN_SCRIPT="${BASE_PATH}/scripts/gpt2/fdd/contra_base.sh"
        CKPT="direct_mlp_t0/fdd/base"
        ;;
    *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac

bash "${TRAIN_SCRIPT}" "${BASE_PATH}" "${MASTER_PORT}" 1 \
    --model-path "${STUDENT_PATH}" \
    --teacher-model-path "${TEACHER_PATH}" \
    --representation-model direct-mlp \
    --velocity-field-path "${MLP_PATH}/direct_state_mlp.pth" \
    --projector-path "${MLP_PATH}/projector.pth" \
    --save "${BASE_PATH}/results/gpt2/train/${CKPT}"

EVAL_PORT=${EVAL_PORT:-2040}
for benchmark in dolly self_inst vicuna sinst; do
    for seed in 10 20 30 40 50; do
        CUDA_VISIBLE_DEVICES=${DEVICE} bash "${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh" \
            "${BASE_PATH}" "${EVAL_PORT}" 1 "${CKPT}" --seed "${seed}" --eval-batch-size 64
    done
done
