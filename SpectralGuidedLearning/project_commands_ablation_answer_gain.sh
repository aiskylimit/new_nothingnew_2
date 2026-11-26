#!/usr/bin/env bash
# E0-E8 answer-gain ablations on four independent GPUs.
#
#   bash project_commands_ablation_answer_gain.sh setup       # scores, labels, weights only
#   bash project_commands_ablation_answer_gain.sh pilot       # train + eval seed 42
#   bash project_commands_ablation_answer_gain.sh all         # setup, then pilot
#   PREDICTABILITY_ARM=e2-predictability-easy \
#     bash project_commands_ablation_answer_gain.sh robustness # seeds 123 and 456
#
# Output (all of it is also tee'd into one master log):
#   screen / logs/answer-gain-<mode>-<time>.log   timestamped START/DONE/FAILED per arm, GPU heartbeat
#   logs/answer-gain-<mode>-<time>-gpu<N>.log     full output of every arm that ran on GPU N
#   logs/<arm>-<track>-<stage>.log                per-stage log written by sgl (the one to open on failure)
#   experiments/answer_gain/results/ablation_summary.md   ONE file: all tables, missing runs, run status
#                                  (+ per_benchmark.csv, seed_summary.csv)
#
# GPU queues (one full-parameter job per GPU):
#   GPU 0: E0, E4, E8       GPU 1: E1, E5
#   GPU 2: E2, E6           GPU 3: E3, E7
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -x "${PROJECT_ENV}/bin/python" ]]; then
  PYTHON="${PROJECT_ENV}/bin/python"
elif [[ -n "${PYTHON:-}" ]]; then
  :
else
  echo "Python env not found: ${PROJECT_ENV}/bin/python (set PROJECT_ENV or PYTHON)" >&2
  exit 1
fi

MODE="${1:-all}"
GPU_IDS=(0 1 2 3)
TRACK="r1-qwen-1.5b-palign"
CONFIG_ROOT="${TRACK}"
RESULTS_OUT="experiments/answer_gain/results"
mkdir -p logs experiments/answer_gain/{scores,figures} "${RESULTS_OUT}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_PREFIX="logs/answer-gain-${MODE}-${STAMP}"
MASTER_LOG="${LOG_PREFIX}.log"
STATUS_FILE="${LOG_PREFIX}.status.tsv"
: > "${STATUS_FILE}"
# From here on everything printed by this script goes to the screen and the master log.
exec > >(tee -a "${MASTER_LOG}") 2>&1

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

RUN_FLAGS=()
[[ "${FORCE:-0}" == 1 ]] && RUN_FLAGS+=(--force)
[[ "${DRY_RUN:-0}" == 1 ]] && RUN_FLAGS+=(--dry-run)

# No network on the GPU box: local model, local benchmarks, Hugging Face / vLLM stay offline.
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
MODEL_NAME="${MODEL_NAME:-${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -f "${MODEL_NAME}/config.json" ]] || { log "model not found: ${MODEL_NAME} (set MODEL_NAME or LOCAL_MODELS_ROOT)"; exit 1; }
  [[ -f references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz ]] || { log "P-ALIGN training data missing"; exit 1; }
  # benchmark dirs as sgl.eval.benchmarks resolves them under BENCH_DATA_ROOT
  for bench in AIME_2024 aime_2025 MATH-500 aimo-validation-amc; do
    [[ -d "${BENCH_DATA_ROOT}/${bench}" ]] || { log "benchmark not found: ${BENCH_DATA_ROOT}/${bench} (set BENCH_DATA_ROOT)"; exit 1; }
  done
fi

COMMON_OVERRIDES=("model.name=${MODEL_NAME}")

log "=== answer-gain ablation: mode=${MODE} ==="
log "commit   : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)$([[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo ' (+uncommitted changes)')"
log "python   : ${PYTHON}"
log "model    : ${MODEL_NAME}"
log "bench    : ${BENCH_DATA_ROOT}"
log "gpus     : ${GPU_IDS[*]}   track: ${TRACK}   force=${FORCE:-0} dry_run=${DRY_RUN:-0}"
log "master log: ${MASTER_LOG}   status: ${STATUS_FILE}"

run_config() {
  local arm="$1" gpu="$2" stages="$3"
  shift 3
  "${PYTHON}" -m sgl.cli run "${CONFIG_ROOT}/${arm}" \
    --stages "${stages}" "${RUN_FLAGS[@]}" \
    "run.gpus=[${gpu}]" "${COMMON_OVERRIDES[@]}" "$@"
}

# One arm (or setup step) on one GPU. Full output goes to $logfile; the screen only gets the
# START / DONE / FAILED event with the log to open. Returns the exit code of the run.
run_arm() {
  local gpu="$1" arm="$2" stages="$3" logfile="$4"
  shift 4
  local run_name="${arm}-${TRACK}" arg start="${SECONDS}" rc=0
  for arg in "$@"; do [[ "${arg}" == run_name=* ]] && run_name="${arg#run_name=}"; done
  log "START  GPU ${gpu}  ${run_name} [${stages}]"
  {
    echo "### $(date '+%F %T') ${run_name} [${stages}] GPU ${gpu}"
    run_config "${arm}" "${gpu}" "${stages}" "$@"
  } >> "${logfile}" 2>&1 || rc=$?
  local elapsed=$(( SECONDS - start ))
  if (( rc == 0 )); then
    log "DONE   GPU ${gpu}  ${run_name} [${stages}] in $(( elapsed / 60 ))m$(( elapsed % 60 ))s"
    printf '%s\t%s\t%s\tok\t%s\t%s\n' "${run_name}" "${stages}" "${gpu}" "${elapsed}" "logs/${run_name}-*.log" >> "${STATUS_FILE}"
  else
    log "FAILED GPU ${gpu}  ${run_name} [${stages}] exit ${rc} after ${elapsed}s"
    log "       stage logs : logs/${run_name}-*.log   |   queue log: ${logfile}"
    log "       last lines of ${logfile}:"
    tail -n 8 "${logfile}" | tr '\r' '\n' | tail -n 8 | sed 's/^/       | /'
    printf '%s\t%s\t%s\tFAILED(%s)\t%s\t%s\n' "${run_name}" "${stages}" "${gpu}" "${rc}" "${elapsed}" "logs/${run_name}-*.log" >> "${STATUS_FILE}"
  fi
  return "${rc}"
}

# Setup steps depend on each other: stop at the first failure.
must() {
  run_arm "$@" || { log "setup aborted: fix the failure above, then re-run (finished stages are skipped)"; exit 1; }
}

wait_jobs() {
  local status=0 pair pid label
  for pair in "$@"; do
    pid="${pair%%:*}"
    label="${pair#*:}"
    wait "${pid}" || { log "FAILED: ${label}"; status=1; }
  done
  return "${status}"
}

# Every N seconds, the last line each GPU log has written (loss / step / vLLM progress).
heartbeat() {
  local interval="${HEARTBEAT_SECS:-600}" gpu line
  while sleep "${interval}" >/dev/null 2>&1; do
    for gpu in "${GPU_IDS[@]}"; do
      [[ -r "${LOG_PREFIX}-gpu${gpu}.log" ]] || continue
      line="$(tr '\r' '\n' < "${LOG_PREFIX}-gpu${gpu}.log" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c1-160 || true)"
      [[ -z "${line}" ]] || log "HEARTBEAT GPU ${gpu}: ${line}"
    done
  done
}

verify_setup() {
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  log "verifying setup artifacts"
  local arm file missing=0
  # E0 is plain SFT, whose `vanilla` stage deliberately writes train-vanilla.jsonl.
  # Every remaining ablation writes an arm-named weighted training file.
  file="data/${TRACK}/train-vanilla.jsonl"
  [[ -s "${file}" ]] || { log "MISSING training data: ${file}"; missing=1; }
  for arm in e1-alg e2-predictability-easy e3-predictability-hard e4-random-assignment \
             e5-alg-no-answer-upweight e6-alg-lambda1 e7-alg-tau1 e8-alg-tau4; do
    file="data/${TRACK}/train-${arm}.jsonl"
    [[ -s "${file}" ]] || { log "MISSING training data: ${file}"; missing=1; }
  done
  (( missing == 0 )) || { log "setup incomplete"; exit 1; }
  "${PYTHON}" - "${TRACK}" <<'PY' | sed 's/^/  /'
import json, sys
from pathlib import Path
track = sys.argv[1]
for path in sorted(Path("data", track).glob("e[1-8]-*-selection-stats.json")):
    stats = next(iter(json.loads(path.read_text())["variants"].values()))
    print(f"{path.name.removesuffix('-selection-stats.json'):28s} max|mass err| {stats['max_abs_mass_error']:.2e}  "
          f"w p10/med/p90 {stats['weight_p10']:.3f}/{stats['weight_median']:.3f}/{stats['weight_p90']:.3f}  "
          f"answer-only steps {stats['answer_only_steps']}")
PY
}

setup_artifacts() {
  local setup_log="${LOG_PREFIX}-setup.log"
  log "setup: shared P-ALIGN data and uniform E0 dataset (GPU ${GPU_IDS[0]})"
  must "${GPU_IDS[0]}" e0-uniform prepare,vanilla "${setup_log}"

  log "setup: frozen-q0 scoring, answer gain on GPU ${GPU_IDS[0]} and predictability on GPU ${GPU_IDS[1]}"
  run_arm "${GPU_IDS[0]}" e1-alg answer_gain "${LOG_PREFIX}-gpu${GPU_IDS[0]}.log" & local gain_pid=$!
  run_arm "${GPU_IDS[1]}" e2-predictability-easy predictability "${LOG_PREFIX}-gpu${GPU_IDS[1]}.log" & local predictability_pid=$!
  wait_jobs "${gain_pid}:answer-gain scoring (GPU ${GPU_IDS[0]})" \
            "${predictability_pid}:predictability scoring (GPU ${GPU_IDS[1]})" \
    || { log "setup aborted: scoring failed"; exit 1; }

  # CPU-only transforms/builds. Each output name is arm-specific, while E1/E5-E8 reuse the
  # exact same immutable answer-gain signal parquet.
  log "setup: score transforms, answer-only labels and weights (CPU work, launched via GPU ${GPU_IDS[0]})"
  must "${GPU_IDS[0]}" e1-alg gain_signal,weights "${setup_log}"
  must "${GPU_IDS[0]}" e2-predictability-easy score_signal,weights "${setup_log}"
  must "${GPU_IDS[0]}" e3-predictability-hard score_transform,score_signal,weights "${setup_log}"
  must "${GPU_IDS[0]}" e4-random-assignment score_transform,gain_signal,weights "${setup_log}"
  must "${GPU_IDS[0]}" e5-alg-no-answer-upweight answer_only,weights "${setup_log}"
  must "${GPU_IDS[0]}" e6-alg-lambda1 weights "${setup_log}"
  must "${GPU_IDS[0]}" e7-alg-tau1 weights "${setup_log}"
  must "${GPU_IDS[0]}" e8-alg-tau4 weights "${setup_log}"
  must "${GPU_IDS[0]}" e4-random-assignment score_analysis "${setup_log}"
  verify_setup
}

# An arm that fails does not stop the arms queued behind it; the queue returns non-zero.
run_queue() {
  local gpu="$1" stages="$2"
  shift 2
  local arm status=0
  for arm in "$@"; do
    run_arm "${gpu}" "${arm}" "${stages}" "${LOG_PREFIX}-gpu${gpu}.log" || status=1
  done
  return "${status}"
}

pilot() {
  local stages="${PILOT_STAGES:-train,eval}"
  run_queue 0 "${stages}" e0-uniform e4-random-assignment e8-alg-tau4 & local p0=$!
  run_queue 1 "${stages}" e1-alg e5-alg-no-answer-upweight & local p1=$!
  run_queue 2 "${stages}" e2-predictability-easy e6-alg-lambda1 & local p2=$!
  run_queue 3 "${stages}" e3-predictability-hard e7-alg-tau1 & local p3=$!
  wait_jobs "${p0}:pilot queue GPU 0" "${p1}:pilot queue GPU 1" \
            "${p2}:pilot queue GPU 2" "${p3}:pilot queue GPU 3"
}

robustness_queue() {
  local gpu="$1" arm="$2" seed status=0
  for seed in 123 456; do
    run_arm "${gpu}" "${arm}" train,eval "${LOG_PREFIX}-gpu${gpu}.log" \
      "train_seed=${seed}" "run_name=${arm}-${TRACK}-s${seed}" || status=1
  done
  return "${status}"
}

robustness() {
  case "${PREDICTABILITY_ARM:-}" in
    e2-predictability-easy|e3-predictability-hard) ;;
    *) log "Set PREDICTABILITY_ARM after the seed-42 pilot to e2-predictability-easy or e3-predictability-hard"
       return 2 ;;
  esac
  robustness_queue 0 e0-uniform & local p0=$!
  robustness_queue 1 e1-alg & local p1=$!
  robustness_queue 2 "${PREDICTABILITY_ARM}" & local p2=$!
  robustness_queue 3 e4-random-assignment & local p3=$!
  wait_jobs "${p0}:robustness E0 GPU 0" "${p1}:robustness E1 GPU 1" \
            "${p2}:robustness predictability GPU 2" "${p3}:robustness E4 GPU 3"
}

# One file with every table: ablation_summary.md (also printed here).
summarize() {
  [[ "${DRY_RUN:-0}" == 1 ]] && return 0
  log "=== summary ==="
  "${PYTHON}" -m sgl.eval.answer_gain_summary --results-dir results --out-dir "${RESULTS_OUT}" \
    --track "${TRACK}" --status-file "${STATUS_FILE}" || log "summary failed (no finished eval yet?)"
  "${PYTHON}" -m sgl.eval.compare --results-dir results > /dev/null 2>&1 || true
}

heartbeat & HEARTBEAT_PID=$!
trap 'kill "${HEARTBEAT_PID}" 2>/dev/null || true' EXIT

rc=0
case "${MODE}" in
  setup) setup_artifacts || rc=$? ;;
  pilot) pilot || rc=$? ;;
  all) setup_artifacts; pilot || rc=$? ;;
  robustness) robustness || rc=$? ;;
  *) echo "usage: $0 {setup|pilot|all|robustness}" >&2; exit 2 ;;
esac
[[ "${MODE}" == setup ]] || summarize
failed="$(grep -c 'FAILED' "${STATUS_FILE}" || true)"
log "=== finished mode=${MODE}: exit ${rc}, ${failed} failed run(s); master log ${MASTER_LOG} ==="
exit "${rc}"
