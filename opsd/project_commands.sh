#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# bash "${PROJECT_ROOT}/setup_env.sh"
source "${ENV_PATH:-/mnt/local/uvenvs/opsd}/bin/activate"

python "${PROJECT_ROOT}/data/prepare_data.py" \
    --raw_root "${PROJECT_ROOT}/data/raw" \
    --output_root "${PROJECT_ROOT}/data/processed" \
    --overwrite

bash "${PROJECT_ROOT}/scripts/run_training.sh" sft 4b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b sft

bash "${PROJECT_ROOT}/scripts/run_training.sh" sft 8b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b sft

bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd 4b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b opsd

bash "${PROJECT_ROOT}/scripts/run_training.sh" opsd 8b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b opsd

bash "${PROJECT_ROOT}/scripts/run_training.sh" grpo 4b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 4b grpo

bash "${PROJECT_ROOT}/scripts/run_training.sh" grpo 8b
bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b grpo