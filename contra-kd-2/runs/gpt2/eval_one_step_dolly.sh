#!/usr/bin/env bash
set -euo pipefail

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
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${ONE_STEP_GPU:-4}

TRAIN_SEED=1024
STUDENT_PATH="${BASE_PATH}/results/gpt2/train/fdd_seed_study/contra_fdd/${TRAIN_SEED}"
TEACHER_PATH="${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}"
DATA_DIR="${BASE_DIR}/data/raw/dolly"
SAVE_PATH="${BASE_PATH}/results/gpt2/eval_main/fdd_seed_study/one_step/${TRAIN_SEED}/dolly"

for file in "${STUDENT_PATH}"/{config.json,args.json,velocity_checkpoint.json,projector.pth,velocity_field.pth} \
    "${TEACHER_PATH}/config.json" "${DATA_DIR}/valid.jsonl"; do
    if [[ ! -s "${file}" ]]; then
        echo "Missing one-step evaluation input: ${file}" >&2
        exit 1
    fi
done

exec torchrun --standalone --nproc-per-node 1 "${BASE_PATH}/evaluate_one_step.py" \
    --model-path "${STUDENT_PATH}" --teacher-model-path "${TEACHER_PATH}" \
    --seed "${TRAIN_SEED}" --data-names dolly --data-dir "${DATA_DIR}" \
    --json-data --max-prompt-length 256 --max-length 512 --num-workers 0 --eval-batch-size 8 \
    --save "${SAVE_PATH}"
