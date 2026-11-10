#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ${1:-} != train ]]; then
    exec bash "${PROJECT_ROOT}/runs/eval_math_code.sh" "$@"
fi

# Must match @PROJECT@ expanded by the provisioner in download.txt.
# PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothingnew_2}"
# BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/contra-kd}"
# ENV_DIR="${ENV_DIR:-/mnt/local/uvenvs/contra}"
# if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
#     echo "Missing environment: ${ENV_DIR}. Provision contra.txt before running offline." >&2
#     exit 1
# fi
# source "${ENV_DIR}/bin/activate"

# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_DISABLE_TELEMETRY=1
# export TOKENIZERS_PARALLELISM=false
# export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
# export RAW_DATA_ROOT="${BASE_DIR}/data/raw"
# export PREPARED_DATA_ROOT="${BASE_DIR}/data/prepared"
# export PROCESSED_DATA_ROOT="${BASE_DIR}/processed_data"
# export OPENWEBTEXT_DATASET_PATH="${RAW_DATA_ROOT}/openwebtext-100k"
# export OPENWEBTEXT_TEXT_PATH="${PREPARED_DATA_ROOT}/openwebtext/data.txt"
# export DOLLY_PROMPT_DATA_DIR="${PREPARED_DATA_ROOT}/dolly"
# # Reuse response pairs from the earlier run. Pair generation is independent of field training.
# export DISTILLM2_DATA_DIR="${DISTILLM2_DATA_DIR:-${BASE_DIR}/data/generated/distillm2/gpt2/t0}"
# export TEACHER_PATH="${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}"
# export STUDENT_PATH="${STUDENT_PATH:-${BASE_DIR}/results/gpt2/train/init/gpt2-base}"

# export CUDA_VISIBLE_DEVICES=4
# export BASE_PORT="${BASE_PORT:-20120}"
# export DATA_PROCESS_WORKERS="${DATA_PROCESS_WORKERS:-4}"

# cd "${PROJECT_ROOT}"
# exec bash "${PROJECT_ROOT}/runs/gpt2/run_contra.sh" "${PROJECT_ROOT}"
