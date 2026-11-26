#!/usr/bin/env bash
# Fill all four GPUs: E9/E10 use 0/1, while the 7B tau=1 tuning sweep begins
# on 2/3.  Once E9/E10 finish, the remaining 7B arms immediately take 0/1.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
LOG_DIR="logs/parallel-e9e10-qwen25-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -f "${PROJECT_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ENV}/bin/activate"
fi
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG_DIR}/master.log") 2>&1

EARLY_ARMS="iwc-gain-l25-t1-c2-b8-lora iwc-gain-l50-t1-c2-b8-lora iwc-gain-l75-t1-c2-b8-lora iwc-gain-l100-t1-c2-b8-lora iwc-gain-l25-t1-c1-b8-lora"
LATE_ARMS="iwc-gain-l50-t1-c1-b8-lora iwc-gain-l75-t1-c1-b8-lora iwc-gain-l25-t1-c3-b8-lora iwc-gain-l50-t1-c3-b8-lora iwc-gain-l75-t1-c3-b8-lora"
ALL_ARMS="${EARLY_ARMS} ${LATE_ARMS}"
{
  echo "started: $(date -Is)"
  echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "project_env: ${PROJECT_ENV}"
  echo "early_arms: ${EARLY_ARMS}"
  echo "late_arms: ${LATE_ARMS}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 || true
} | tee "${LOG_DIR}/manifest.txt"

# Shared 7B artifact creation needs only GPU 2 and can overlap E9/E10 setup.
GPU_IDS="2" ARMS_OVERRIDE="${EARLY_ARMS}" MODE=setup \
  bash project_commands_tune_qwen25_7b_tau1_b8.sh 2>&1 | tee "${LOG_DIR}/7b-setup.log" & setup_pid=$!
bash project_commands_ablation_answer_gain_norm.sh 2>&1 | tee "${LOG_DIR}/e9-e10.log" & norm_pid=$!

wait "${setup_pid}"
GPU_IDS="2 3" ARMS_OVERRIDE="${EARLY_ARMS}" MODE=run \
  bash project_commands_tune_qwen25_7b_tau1_b8.sh 2>&1 | tee "${LOG_DIR}/7b-early.log" & early_pid=$!

wait "${norm_pid}"
GPU_IDS="0 1" ARMS_OVERRIDE="${LATE_ARMS}" MODE=run \
  bash project_commands_tune_qwen25_7b_tau1_b8.sh 2>&1 | tee "${LOG_DIR}/7b-late.log" & late_pid=$!

wait "${early_pid}" "${late_pid}"

echo "=== E9/E10 normalization report ==="
"${PYTHON}" -m sgl.eval.answer_gain_summary --results-dir results --out-dir results \
  --track r1-qwen-1.5b-palign | tee "${LOG_DIR}/normalization-summary.log"
echo "=== Qwen2.5-7B tuning report ==="
"${PYTHON}" -m sgl.eval.qwen25_tuning_summary --results-dir results \
  --out-dir results/qwen25-tuning-tau1-b8 --arms ${ALL_ARMS} | tee "${LOG_DIR}/qwen25-summary.log"
echo "completed: $(date -Is); master log: ${LOG_DIR}/master.log"
