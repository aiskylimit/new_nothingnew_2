#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothingnew_2}"
export BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/contra-kd}"
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export STUDENT_PATH="${STUDENT_PATH:-${BASE_DIR}/results/gpt2/train/init/gpt2-base}"
export TEACHER_PATH="${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}"
export PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-${BASE_DIR}/processed_data}"
export RAW_DATA_ROOT="${RAW_DATA_ROOT:-${BASE_DIR}/data/raw}"

cd "${PROJECT_ROOT}"
case "${1:-loss-ramp}" in
    loss-ramp)
        exec bash "${PROJECT_ROOT}/runs/gpt2/run_loss_ramp_ablation.sh" "${PROJECT_ROOT}"
        ;;
    direct-mlp-fdd)
        rerun_log_dir="${PROJECT_ROOT}/logs/gpt2_direct_mlp_ablation"
        mkdir -p "${rerun_log_dir}"
        rerun_log="${rerun_log_dir}/fdd_rerun_$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
        echo "Rerunning direct MLP FDD on GPU 7; log: ${rerun_log}"
        export CUDA_VISIBLE_DEVICES=7
        export PET_STANDALONE=1
        export EVAL_PORT="${EVAL_PORT:-$(( ${BASE_PORT:-20120} + 21 ))}"
        exec bash "${PROJECT_ROOT}/runs/gpt2/direct_mlp.sh" \
            "${PROJECT_ROOT}" "$(( ${BASE_PORT:-20120} + 11 ))" fdd \
            > "${rerun_log}" 2>&1
        ;;
    *)
        echo "Usage: bash project_commands.sh [loss-ramp|direct-mlp-fdd]" >&2
        exit 2
        ;;
esac
