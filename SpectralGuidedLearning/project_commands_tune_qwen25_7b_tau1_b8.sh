#!/usr/bin/env bash
# Ten Qwen2.5-7B answer-gain tuning arms: lambda {.25,.5,.75,1},
# clip {1,2,3}, tau=1, batch 8.  c=2 has the complete lambda sweep; c=1/3
# probe the lower/higher gain contrast at lambda {.25,.5,.75}.
# One GPU per job.  MODE=setup materializes shared artifacts; MODE=run consumes
# them.  GPU_IDS and ARMS_OVERRIDE allow an outer scheduler to fill GPUs as
# they become free without rebuilding or racing the shared artifacts.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -f "${PROJECT_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ENV}/bin/activate"
fi
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

TRACK="qwen25-7b-palign"
MODEL_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${MODEL_NAME:-${MODEL_ROOT}/Qwen2.5-7B-Instruct}"
LOG_DIR="logs/qwen25-7b-tau1-b8-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"
MODE="${MODE:-all}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
read -r -a GPUS <<< "${GPU_IDS}"
ARMS=(
  iwc-gain-l25-t1-c2-b8-lora iwc-gain-l50-t1-c2-b8-lora
  iwc-gain-l75-t1-c2-b8-lora iwc-gain-l100-t1-c2-b8-lora
  iwc-gain-l25-t1-c1-b8-lora iwc-gain-l50-t1-c1-b8-lora
  iwc-gain-l75-t1-c1-b8-lora iwc-gain-l25-t1-c3-b8-lora
  iwc-gain-l50-t1-c3-b8-lora iwc-gain-l75-t1-c3-b8-lora
)
if [[ -n "${ARMS_OVERRIDE:-}" ]]; then
  read -r -a ARMS <<< "${ARMS_OVERRIDE}"
fi
(( ${#GPUS[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU" >&2; exit 2; }
(( ${#ARMS[@]} > 0 )) || { echo "ARMS_OVERRIDE must contain at least one arm" >&2; exit 2; }

run() { "${PYTHON}" -m sgl.cli run "${TRACK}/$1" "${@:2}" "model.name=${MODEL_NAME}"; }

setup() {
  local setup_gpu="${GPUS[0]}" arm
  # The first arm materializes shared frozen-model artifacts once.  Each arm
  # receives an isolated weighted JSONL via train_variant=${name}.
  run "${ARMS[0]}" --stages prepare,capture,answer_gain,gain_signal "run.gpus=[${setup_gpu}]" \
    2>&1 | tee "${LOG_DIR}/setup.log"
  for arm in "${ARMS[@]}"; do
    run "${arm}" --stages weights "run.gpus=[${setup_gpu}]" 2>&1 | tee "${LOG_DIR}/${arm}-weights.log"
  done
}

queue() {
  local gpu="$1"; shift
  local arm
  for arm in "$@"; do
    run "${arm}" --stages train,eval "run.gpus=[${gpu}]" 2>&1 | tee "${LOG_DIR}/${arm}.log"
  done
}
run_queues() {
  local index gpu position=0 pids=()
  for gpu in "${GPUS[@]}"; do
    local assigned=()
    for ((index = position; index < ${#ARMS[@]}; index += ${#GPUS[@]})); do
      assigned+=("${ARMS[index]}")
    done
    queue "${gpu}" "${assigned[@]}" & pids+=("$!")
    ((position += 1))
  done
  wait "${pids[@]}"
}

case "${MODE}" in
  setup) setup ;;
  run) run_queues ;;
  all) setup; run_queues ;;
  *) echo "MODE must be setup, run, or all" >&2; exit 2 ;;
esac
