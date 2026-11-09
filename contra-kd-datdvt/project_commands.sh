#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Share the original project's environment, data and checkpoints.
PROJECT_NAME="${PROJECT_NAME:-aiskylimit_new_nothingnew_2}"
BASE_DIR="${BASE_DIR:-/mnt/local/${PROJECT_NAME}/contra-kd}"
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
# Physical GPU 6 becomes logical cuda:0. No torchrun or training is involved.
export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "${PROJECT_ROOT}"
exec python "${PROJECT_ROOT}/tools/evaluate_layer_changes.py" \
    --teacher "${TEACHER_PATH:-${BASE_DIR}/results/gpt2/train/sft/gpt2-xlarge}" \
    --student "fdd=${FDD_PATH:-${BASE_DIR}/results/gpt2/train/fdd/base}" \
    --student "contra_fdd=${CONTRA_FDD_PATH:-${BASE_DIR}/results/gpt2/train/contra/fdd/base}" \
    --student "distillm=${DISTILLM_PATH:-${BASE_DIR}/results/gpt2/train/distillm/base}" \
    --student "contra_distillm=${CONTRA_DISTILLM_PATH:-${BASE_DIR}/results/gpt2/train/contra/distillm/base}" \
    --compare fdd=contra_fdd --compare distillm=contra_distillm \
    --data "dolly=${BASE_DIR}/data/raw/dolly/valid.jsonl" \
    --data "self_inst=${BASE_DIR}/data/raw/self-inst/valid.jsonl" \
    --output-root "${OUTPUT_ROOT:-${PROJECT_ROOT}/results/layer_changes}" \
    --max-length 512 --max-prompt-length 256 --token-chunk-size 16 \
    --bootstrap-samples 2000 --seed 42 "$@"
