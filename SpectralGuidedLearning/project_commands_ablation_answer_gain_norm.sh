#!/usr/bin/env bash
# Budget-normalization ablation after E0-E8: E9 global corpus norm vs E10 no norm.
# Uses the immutable E7 score cache and launches one single-GPU job per arm.
#
#   PROJECT_ENV=/mnt/local/uvenvs/spectral_guided_learning \
#   bash project_commands_ablation_answer_gain_norm.sh
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
PYTHON="${PYTHON:-python}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -f "${PROJECT_ENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ENV}/bin/activate"
  PYTHON="${PYTHON:-python}"
fi
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

TRACK="r1-qwen-1.5b-palign"
MODEL_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${MODEL_NAME:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
LOG_DIR="logs/normalization-ablation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

run() {
  "${PYTHON}" -m sgl.cli run "$@" "model.name=${MODEL_NAME}"
}

# Scores and tokenized traces are shared and immutable.  E9 creates its own
# globally normalized dataset; E10 consumes that same score parquet separately.
run "${TRACK}/e9-alg-global-norm" --stages prepare,answer_gain,gain_signal,weights \
  "run.gpus=[0]" 2>&1 | tee "${LOG_DIR}/setup-e9.log"
run "${TRACK}/e10-alg-no-norm" --stages weights "run.gpus=[0]" \
  2>&1 | tee "${LOG_DIR}/setup-e10.log"

# These 1.5B arms have low VRAM demand.  Each keeps the prescribed effective
# batch 32 with one rank (accumulation = 32) while both arms run concurrently.
(run "${TRACK}/e9-alg-global-norm" --stages train,eval "run.gpus=[0]" \
  2>&1 | tee "${LOG_DIR}/e9.log") & p9=$!
(run "${TRACK}/e10-alg-no-norm" --stages train,eval "run.gpus=[1]" \
  2>&1 | tee "${LOG_DIR}/e10.log") & p10=$!
wait "${p9}"
wait "${p10}"

"${PYTHON}" -m sgl.eval.answer_gain_summary --results-dir results --out-dir results \
  --track "${TRACK}" 2>&1 | tee "${LOG_DIR}/summary.log"
