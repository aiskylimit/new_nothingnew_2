#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-coarse-dynamics}" in
    coarse-dynamics) exec bash "${PROJECT_ROOT}/runs/gpt2/eval_coarse_dynamics.sh" "${PROJECT_ROOT}" ;;
    export) exec python3 "${PROJECT_ROOT}/tools/export_eval_results.py" "${PROJECT_ROOT}/results/gpt2/eval_main" ;;
    fdd-batch-size-sweep)
        bash "${PROJECT_ROOT}/runs/gpt2/eval_fdd_batch_sizes.sh" "${PROJECT_ROOT}"
        exec bash "${PROJECT_ROOT}/project_commands.sh" export
        ;;
    fdd-seed-study) ;;
    one-step-dolly) exec bash "${PROJECT_ROOT}/runs/gpt2/eval_one_step_dolly.sh" "${PROJECT_ROOT}" ;;
    *) echo "Usage: bash project_commands.sh [coarse-dynamics|export|fdd-batch-size-sweep|fdd-seed-study|one-step-dolly]" >&2; exit 2 ;;
esac

# Must match @PROJECT@ expanded by the provisioner in download.txt.
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
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${BASE_DIR}/.cache/huggingface}"
export RAW_DATA_ROOT="${RAW_DATA_ROOT:-${BASE_DIR}/data/raw}"
export PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-${BASE_DIR}/data/prepared}"
export PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${BASE_DIR}/processed_data}"
export OPENWEBTEXT_DATASET_PATH="${OPENWEBTEXT_DATASET_PATH:-${RAW_DATA_ROOT}/openwebtext-100k}"
export TEACHER_PATH="${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}"
export STUDENT_PATH="${STUDENT_PATH:-${BASE_DIR}/results/gpt2/train/init/gpt2-base}"

export BASE_PORT="${BASE_PORT:-20120}"
export DATA_PROCESS_WORKERS="${DATA_PROCESS_WORKERS:-4}"

bash "${PROJECT_ROOT}/runs/gpt2/run_fdd_seed_study.sh" "${PROJECT_ROOT}" additional-seeds
exec bash "${PROJECT_ROOT}/project_commands.sh" export
