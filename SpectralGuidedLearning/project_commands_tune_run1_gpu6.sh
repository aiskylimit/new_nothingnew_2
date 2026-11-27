#!/usr/bin/env bash
# Only the fourth arm of project_commands_tune_run1.sh (ALERD tau 1, lambda 0.7, s1K-1.1, seed 42), so it can
# start while the other three are running. Needs the traces and gains that script built.
#   bash project_commands_tune_run1_gpu6.sh [--dry-run]      GPU=6 (default)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
MODEL_PATH="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}/DeepSeek-R1-Distill-Qwen-1.5B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

OUT=rebuttal/tune
GPU="${GPU:-6}"
if [[ " $* " != *" --dry-run "* ]]; then
  source "${PROJECT_ENV}/bin/activate"
  [[ -f data/r1-qwen-1.5b-s1k/signals/signal-answer-gain.parquet ]] ||
    { echo "gains missing: run project_commands_tune_run1.sh first" >&2; exit 1; }
  mkdir -p "${OUT}/logs"
  exec > >(tee -a "${OUT}/logs/driver-run1-gpu6-$(date +%Y%m%d-%H%M%S).log") 2>&1
  nvidia-smi -i "${GPU}" --query-gpu=index,name,memory.total,memory.used --format=csv
fi

python -m sgl.cli run r1-qwen-1.5b-s1k/iwc-gain-l05 --stages weights,train,eval "run.gpus=[${GPU}]" \
  "model.name=${MODEL_PATH}" "dataset=${DATASET_NAME}" "results_dir=${OUT}/results" "run.log_dir=${OUT}/logs" \
  "run.efficiency_dir=${OUT}/efficiency" name=iwc-gain-t1-l07 train_variant=iwc-gain-t1-l07 \
  iwc.temperature=1.0 iwc.interpolation=0.7 "$@"
[[ " $* " == *" --dry-run "* ]] || python scripts/report_run1.py
