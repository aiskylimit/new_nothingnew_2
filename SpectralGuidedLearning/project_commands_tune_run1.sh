#!/usr/bin/env bash
# Run 1 sweep (1.5B student, s1K-1.1, seed 42): four ALERD allocation strengths, one per GPU.
#   GPU 0  tau 2, lambda 0.5   iwc-gain-t2-l05
#   GPU 1  tau 1, lambda 0.3   iwc-gain-t1-l03
#   GPU 6  tau 1, lambda 0.7   iwc-gain-t1-l07
#   GPU 7  tau 2, lambda 1.0   iwc-gain-t2-l10
# Already run: tau 1 lambda 0.5 (paper) and tau 0.5 lambda 0.5.
# A sensitivity sweep scored directly on the test benchmarks, with no dev set: report all of them, and keep
# tau 1 lambda 0.5 (the paper's, fixed beforehand) as the headline number.
#   bash project_commands_tune_run1.sh              # train + eval + table
#   bash project_commands_tune_run1.sh --dry-run    # print every stage command, run nothing
#   GPUS="0 1 6 7" (default), in the order above. Finished stages are skipped, so relaunch after a crash.
# Outputs: rebuttal/tune/results/, rebuttal/tune/logs/.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1
ARGS=("$@")
read -ra G <<< "${GPUS:-0 1 6 7}"
[[ "${#G[@]}" == 4 ]] || { echo "GPUS needs 4 ids, got: ${G[*]}" >&2; exit 2; }

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
MODEL_PATH="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}/DeepSeek-R1-Distill-Qwen-1.5B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

OUT=rebuttal/tune
CONFIG=r1-qwen-1.5b-s1k/iwc-gain-l05

if [[ "${DRY_RUN}" == 0 ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV}" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"
  missing=0
  for path in "${MODEL_PATH}" "${DATASET_NAME}" "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
    [[ -d "${path}" ]] || { echo "missing: ${path} (see download.txt)" >&2; missing=1; }
  done
  [[ "${missing}" == 0 ]] || exit 1
  mkdir -p "${OUT}/logs"
  exec > >(tee -a "${OUT}/logs/driver-run1-$(date +%Y%m%d-%H%M%S).log") 2>&1
  nvidia-smi -i "$(IFS=,; echo "${G[*]}")" --query-gpu=index,name,memory.total,memory.used --format=csv
fi

# sgl_run <gpu> <stages> [overrides]
sgl_run() {
  local gpu="$1" stages="$2"
  python -m sgl.cli run "${CONFIG}" --stages "${stages}" "run.gpus=[${gpu}]" "model.name=${MODEL_PATH}" \
    "dataset=${DATASET_NAME}" "results_dir=${OUT}/results" "run.log_dir=${OUT}/logs" \
    "run.efficiency_dir=${OUT}/efficiency" "${@:3}" ${ARGS[@]+"${ARGS[@]}"}
}

# arm <gpu> <arm> <tau> <lambda>
arm() {
  sgl_run "$1" weights,train,eval "name=$2" "train_variant=$2" "iwc.temperature=$3" "iwc.interpolation=$4"
}

job0() { arm "$1" iwc-gain-t2-l05 2.0 0.5; }
job1() { arm "$1" iwc-gain-t1-l03 1.0 0.3; }
job2() { arm "$1" iwc-gain-t1-l07 1.0 0.7; }
job3() { arm "$1" iwc-gain-t2-l10 2.0 1.0; }

# the traces and gains are shared by all arms: build them once (skipped if run 1 left them) before starting
sgl_run "${G[0]}" prepare,capture,answer_gain,gain_signal

if [[ "${DRY_RUN}" == 1 ]]; then
  for i in 0 1 2 3; do "job${i}" "${G[$i]}"; done
  exit 0
fi

trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops all four
pids=()
for i in 0 1 2 3; do
  ( "job${i}" "${G[$i]}" 2>&1 | sed -u "s/^/[gpu${G[$i]}] /" ) & pids+=($!)
done
failed=()
for i in 0 1 2 3; do wait "${pids[$i]}" || failed+=("gpu${G[$i]}"); done
python scripts/report_run1.py
[[ "${#failed[@]}" == 0 ]] || { echo "FAILED: ${failed[*]} (see ${OUT}/logs/)" >&2; exit 1; }
