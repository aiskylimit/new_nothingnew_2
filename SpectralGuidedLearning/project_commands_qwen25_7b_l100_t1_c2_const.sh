#!/usr/bin/env bash
# Qwen2.5-7B answer-gain arm lambda=1.0, tau=1.0, clip=2, batch 8, constant lr after warmup, GPU 0.
# Runs everything with no arguments: shared artifacts (skipped when they already exist), this arm's
# weighted JSONL, then train + eval.
#   bash project_commands_qwen25_7b_l100_t1_c2_const.sh
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
PROJECT_ENV="/mnt/local/uvenvs/spectral_guided_learning"
if [[ -f "${PROJECT_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

CONFIG="qwen25-7b-palign/iwc-gain-l100-t1-c2-b8-const-lora"
MODEL_NAME="/mnt/local/_models/aiskylimit_new_nothingnew_2/Qwen2.5-7B-Instruct"
LOG_DIR="logs/qwen25-7b-l100-t1-c2-const-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

python -m sgl.cli run "${CONFIG}" --stages prepare,capture,answer_gain,gain_signal,weights,train,eval \
  "run.gpus=[0]" "model.name=${MODEL_NAME}" 2>&1 | tee "${LOG_DIR}/run.log"
