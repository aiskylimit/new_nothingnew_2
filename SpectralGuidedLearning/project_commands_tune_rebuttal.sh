#!/usr/bin/env bash
# Three rebuttal variants of runs 1, 10 and 13 (1.5B student), one config and one seed (42) each, one per GPU:
#   GPU 0  run 1   ALERD, s1K-1.1, tau 0.5 (before: 1.0), lambda 0.5
#   GPU 1  run 10  answer-steps (mention), 966 P-ALIGN traces (before: s1K-1.1)
#   GPU 6  run 13  ASFT, 966 traces, KL lambda 0.1 (before: 0.05)
# These do not replace the first runs: report both. Single configs scored directly on the test benchmarks.
#   bash project_commands_tune_rebuttal.sh              # train + eval, wait for all three
#   bash project_commands_tune_rebuttal.sh --dry-run    # print every stage command, run nothing
#   GPUS="0 1 6" (default), in the order above. Finished stages are skipped, so relaunch after a crash.
# Outputs: rebuttal/tune/results/ (scores, comparison table), rebuttal/tune/logs/.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1
ARGS=("$@")
read -ra G <<< "${GPUS:-0 1 6}"
[[ "${#G[@]}" == 3 ]] || { echo "GPUS needs 3 ids, got: ${G[*]}" >&2; exit 2; }

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

if [[ "${DRY_RUN}" == 0 ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV}" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"
  missing=0
  for path in "${MODEL_PATH}" "${DATASET_NAME}" "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
    [[ -d "${path}" ]] || { echo "missing: ${path} (see download.txt)" >&2; missing=1; }
  done
  [[ "${missing}" == 0 ]] || exit 1
  mkdir -p "${OUT}/logs"
  exec > >(tee -a "${OUT}/logs/driver-$(date +%Y%m%d-%H%M%S).log") 2>&1
  nvidia-smi -i "${G[0]},${G[1]},${G[2]}" --query-gpu=index,name,memory.total,memory.used --format=csv
fi

# sgl_run <gpu> <config> <stages> [overrides]
sgl_run() {
  local gpu="$1" config="$2" stages="$3" data=()
  [[ "${config}" == *-s1k/* ]] && data=("dataset=${DATASET_NAME}")
  python -m sgl.cli run "${config}" --stages "${stages}" "run.gpus=[${gpu}]" "model.name=${MODEL_PATH}" \
    "results_dir=${OUT}/results" "run.log_dir=${OUT}/logs" "run.efficiency_dir=${OUT}/efficiency" \
    ${data[@]+"${data[@]}"} "${@:4}" ${ARGS[@]+"${ARGS[@]}"}
}

# its own name and dataset file, so the tau 1.0 run 1 stays as it is
run1() {
  sgl_run "$1" r1-qwen-1.5b-s1k/iwc-gain-l05 prepare,capture,answer_gain,gain_signal,weights,train,eval \
    name=iwc-gain-t05-l05 iwc.temperature=0.5 train_variant=iwc-gain-t05-l05
}
run10() { sgl_run "$1" r1-qwen-1.5b-palign/answer-steps-only prepare,answer_mask,train,eval; }
run13() { sgl_run "$1" r1-qwen-1.5b-palign/sft-asft train,eval name=sft-asft-kl01 asft_kl_lambda=0.1; }

# runs 10 and 13 share data/r1-qwen-1.5b-palign/: build it once, before they start side by side
sgl_run "${G[1]}" r1-qwen-1.5b-palign/sft-asft prepare,vanilla

if [[ "${DRY_RUN}" == 1 ]]; then
  run1 "${G[0]}"; run10 "${G[1]}"; run13 "${G[2]}"
  exit 0
fi

trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops all three
pids=() names=(run1 run10 run13)
for i in 0 1 2; do
  ( "${names[$i]}" "${G[$i]}" 2>&1 | sed -u "s/^/[gpu${G[$i]} ${names[$i]}] /" ) & pids+=($!)
done
failed=()
for i in 0 1 2; do wait "${pids[$i]}" || failed+=("${names[$i]}"); done
python -m sgl.eval.compare --results-dir "${OUT}/results"
[[ "${#failed[@]}" == 0 ]] || { echo "FAILED: ${failed[*]} (see ${OUT}/logs/)" >&2; exit 1; }
