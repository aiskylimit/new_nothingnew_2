#!/usr/bin/env bash
# E0-E8 answer-gain ablations on four independent GPUs.
#
#   bash project_commands_ablation_answer_gain.sh setup       # scores, labels, weights only
#   bash project_commands_ablation_answer_gain.sh pilot       # train + eval seed 42
#   bash project_commands_ablation_answer_gain.sh all         # setup, then pilot
#   PREDICTABILITY_ARM=e2-predictability-easy \
#     bash project_commands_ablation_answer_gain.sh robustness # seeds 123 and 456
#
# GPU queues (one full-parameter job per GPU):
#   GPU 2: E0, E4, E8       GPU 3: E1, E5
#   GPU 4: E2, E6           GPU 5: E3, E7
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE}/.." && pwd)/iwc}"
if [[ -x "${PROJECT_ENV}/bin/python" ]]; then
  PYTHON="${PROJECT_ENV}/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

MODE="${1:-all}"
GPU_IDS=(2 3 4 5)
TRACK="r1-qwen-1.5b-palign"
CONFIG_ROOT="${TRACK}"
mkdir -p logs experiments/answer_gain/{scores,results,figures}

RUN_FLAGS=()
[[ "${FORCE:-0}" == 1 ]] && RUN_FLAGS+=(--force)
[[ "${DRY_RUN:-0}" == 1 ]] && RUN_FLAGS+=(--dry-run)

COMMON_OVERRIDES=()
[[ -z "${MODEL_NAME:-}" ]] || COMMON_OVERRIDES+=("model.name=${MODEL_NAME}")

run_config() {
  local arm="$1" gpu="$2" stages="$3"
  shift 3
  "${PYTHON}" -m sgl.cli run "${CONFIG_ROOT}/${arm}" \
    --stages "${stages}" "${RUN_FLAGS[@]}" \
    "run.gpus=[${gpu}]" "${COMMON_OVERRIDES[@]}" "$@"
}

wait_jobs() {
  local status=0 pair pid label
  for pair in "$@"; do
    pid="${pair%%:*}"
    label="${pair#*:}"
    wait "${pid}" || { echo "FAILED: ${label}" >&2; status=1; }
  done
  return "${status}"
}

setup_artifacts() {
  echo "Preparing the shared P-ALIGN data and uniform E0 dataset on GPU ${GPU_IDS[0]}"
  run_config e0-uniform "${GPU_IDS[0]}" prepare,vanilla

  echo "Computing frozen-q0 answer gain on GPU ${GPU_IDS[0]} and predictability on GPU ${GPU_IDS[1]}"
  run_config e1-alg "${GPU_IDS[0]}" answer_gain > logs/answer-gain-score-gpu2.log 2>&1 &
  local gain_pid=$!
  run_config e2-predictability-easy "${GPU_IDS[1]}" predictability > logs/predictability-score-gpu3.log 2>&1 &
  local predictability_pid=$!
  wait_jobs "${gain_pid}:answer-gain scoring (GPU 2)" \
            "${predictability_pid}:predictability scoring (GPU 3)"

  # CPU-only transforms/builds. Each output name is arm-specific, while E1/E5-E8 reuse the
  # exact same immutable answer-gain signal parquet.
  run_config e1-alg "${GPU_IDS[0]}" gain_signal,weights
  run_config e2-predictability-easy "${GPU_IDS[0]}" score_signal,weights
  run_config e3-predictability-hard "${GPU_IDS[0]}" score_transform,score_signal,weights
  run_config e4-random-assignment "${GPU_IDS[0]}" score_transform,gain_signal,weights
  run_config e5-alg-no-answer-upweight "${GPU_IDS[0]}" answer_only,weights
  run_config e6-alg-lambda1 "${GPU_IDS[0]}" weights
  run_config e7-alg-tau1 "${GPU_IDS[0]}" weights
  run_config e8-alg-tau4 "${GPU_IDS[0]}" weights
  run_config e4-random-assignment "${GPU_IDS[0]}" score_analysis
}

run_queue() {
  local gpu="$1" stages="$2"
  shift 2
  local arm
  for arm in "$@"; do
    echo "GPU ${gpu}: ${arm} (${stages})"
    run_config "${arm}" "${gpu}" "${stages}"
  done
}

pilot() {
  local stages="${PILOT_STAGES:-train,eval}"
  run_queue 2 "${stages}" e0-uniform e4-random-assignment e8-alg-tau4 \
    > logs/answer-gain-ablation-gpu2.log 2>&1 & local p2=$!
  run_queue 3 "${stages}" e1-alg e5-alg-no-answer-upweight \
    > logs/answer-gain-ablation-gpu3.log 2>&1 & local p3=$!
  run_queue 4 "${stages}" e2-predictability-easy e6-alg-lambda1 \
    > logs/answer-gain-ablation-gpu4.log 2>&1 & local p4=$!
  run_queue 5 "${stages}" e3-predictability-hard e7-alg-tau1 \
    > logs/answer-gain-ablation-gpu5.log 2>&1 & local p5=$!
  wait_jobs "${p2}:pilot queue GPU 2" "${p3}:pilot queue GPU 3" \
            "${p4}:pilot queue GPU 4" "${p5}:pilot queue GPU 5"
  [[ "${DRY_RUN:-0}" == 1 ]] || "${PYTHON}" -m sgl.eval.compare --results-dir results
}

robustness_queue() {
  local gpu="$1" arm="$2" seed run_name
  for seed in 123 456; do
    run_name="${arm}-${TRACK}-s${seed}"
    run_config "${arm}" "${gpu}" train,eval \
      "train_seed=${seed}" "run_name=${run_name}"
  done
}

robustness() {
  case "${PREDICTABILITY_ARM:-}" in
    e2-predictability-easy|e3-predictability-hard) ;;
    *) echo "Set PREDICTABILITY_ARM after the seed-42 pilot to e2-predictability-easy or e3-predictability-hard" >&2
       return 2 ;;
  esac
  robustness_queue 2 e0-uniform > logs/answer-gain-robustness-gpu2.log 2>&1 & local p2=$!
  robustness_queue 3 e1-alg > logs/answer-gain-robustness-gpu3.log 2>&1 & local p3=$!
  robustness_queue 4 "${PREDICTABILITY_ARM}" > logs/answer-gain-robustness-gpu4.log 2>&1 & local p4=$!
  robustness_queue 5 e4-random-assignment > logs/answer-gain-robustness-gpu5.log 2>&1 & local p5=$!
  wait_jobs "${p2}:robustness E0 GPU 2" "${p3}:robustness E1 GPU 3" \
            "${p4}:robustness predictability GPU 4" "${p5}:robustness E4 GPU 5"
  [[ "${DRY_RUN:-0}" == 1 ]] || "${PYTHON}" -m sgl.eval.compare --results-dir results
}

case "${MODE}" in
  setup) setup_artifacts ;;
  pilot) pilot ;;
  all) setup_artifacts; pilot ;;
  robustness) robustness ;;
  *) echo "usage: $0 {setup|pilot|all|robustness}" >&2; exit 2 ;;
esac
