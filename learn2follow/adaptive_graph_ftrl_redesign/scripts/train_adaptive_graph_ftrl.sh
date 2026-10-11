#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    'Usage: [NAME=value ...] bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh [training CLI options]' \
    'Defaults:' \
    '  DUEL_PROBLEMS=8 HELDOUT_PROBLEMS=128 WARMUP_STEPS=5 MIN_VALID_CONTROLS=8' \
    '  MAX_STEPS=75 SAVE_EVERY=25 EVAL_STEPS=25,50,75 EVAL_DATASETS=aime24,aime25' \
    '  CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_SEED=42 EVAL_SEED=42' \
    '  EVAL_GPU_MEMORY_UTILIZATION=0.9' \
    '  RUN_ROOT=$HOME/dsd/outputs RUN_DIR=<new unique directory under RUN_ROOT>' \
    '  EVAL_WORK_DIR=<short unique sibling .runtime directory> PYTHON_BIN=python' \
    '  CONFIG=adaptive_graph_ftrl_redesign/configs/hard_experiment.yaml' \
    'Optional overrides (unset = use YAML/default config):' \
    '  FTRL_RHO FTRL_ETA FTRL_MIN_WEIGHT DELTA_RMS_DECAY DELTA_RMS_FLOOR DELTA_SCALED_CLIP' \
    '  WINDOW_STEPS ACTOR_WARMUP_WINDOWS CONTROLLER_LR EXPOSURE_TEMPERATURE LEARNING_RATE' \
    '  LAMBDA_GT GAMMA_DISCOUNT' \
    'Equivalent training CLI options are accepted, e.g. --ftrl-rho=0.8.' \
    'CLI options take precedence over these optional environment overrides.' \
    'FTRL_ETA and FTRL_MIN_WEIGHT affect soft selection only, not hard selection.' \
    'DUEL_PROBLEMS controls heldout scoring of virtual branches, not the main gradient batch.' \
    'EVAL_STEPS is generated from SAVE_EVERY and MAX_STEPS when unset.' \
    'Teacher selection defaults to hard, MASK includes the final answer.' \
    'Generated names include resolved hyperparameters and seeds; full settings are saved in config.yaml.'
  exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DUEL_PROBLEMS="${DUEL_PROBLEMS:-8}"
HELDOUT_PROBLEMS="${HELDOUT_PROBLEMS:-128}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MIN_VALID_CONTROLS="${MIN_VALID_CONTROLS:-8}"
export MAX_STEPS="${MAX_STEPS:-75}"
SAVE_EVERY="${SAVE_EVERY:-25}"
for name in DUEL_PROBLEMS HELDOUT_PROBLEMS WARMUP_STEPS MIN_VALID_CONTROLS MAX_STEPS SAVE_EVERY; do
  if [[ ! "${!name}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "$name must be a nonnegative integer without leading zeros" >&2
    exit 2
  fi
done
if (( MIN_VALID_CONTROLS < 1 || MIN_VALID_CONTROLS > DUEL_PROBLEMS || DUEL_PROBLEMS > HELDOUT_PROBLEMS || MAX_STEPS < 1 || SAVE_EVERY < 1 || WARMUP_STEPS > MAX_STEPS )); then
  echo "Require 1 <= MIN_VALID_CONTROLS <= DUEL_PROBLEMS <= HELDOUT_PROBLEMS, positive MAX_STEPS/SAVE_EVERY, and WARMUP_STEPS <= MAX_STEPS" >&2
  exit 2
fi

if [[ -z "${EVAL_STEPS:-}" ]]; then
  EVAL_STEPS=""
  for ((step=SAVE_EVERY; step<MAX_STEPS; step+=SAVE_EVERY)); do
    EVAL_STEPS+="${step},"
  done
  EVAL_STEPS+="$MAX_STEPS"
fi
if [[ ! "$EVAL_STEPS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "EVAL_STEPS must be comma-separated positive checkpoint steps" >&2
  exit 2
fi
IFS=',' read -r -a checkpoints <<< "$EVAL_STEPS"
for step in "${checkpoints[@]}"; do
  if (( step > MAX_STEPS || (step != MAX_STEPS && step % SAVE_EVERY != 0) )); then
    echo "Checkpoint $step will not be saved; change EVAL_STEPS or SAVE_EVERY" >&2
    exit 2
  fi
done
export EVAL_STEPS
export EVAL_DATASETS="${EVAL_DATASETS:-aime24,aime25}"
if [[ ! "$EVAL_DATASETS" =~ ^(aime2[456]|hmmt25)(,(aime2[456]|hmmt25))*$ ]]; then
  echo "EVAL_DATASETS must contain comma-separated aime24, aime25, aime26 or hmmt25" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAIN_SEED="${TRAIN_SEED:-42}" EVAL_SEED="${EVAL_SEED:-42}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
export CONFIG="${CONFIG:-adaptive_graph_ftrl_redesign/configs/hard_experiment.yaml}"
hyperparameter_args=()
for setting in \
  FTRL_RHO:ftrl-rho FTRL_ETA:ftrl-eta FTRL_MIN_WEIGHT:ftrl-min-weight \
  DELTA_RMS_DECAY:delta-rms-decay DELTA_RMS_FLOOR:delta-rms-floor DELTA_SCALED_CLIP:delta-scaled-clip \
  WINDOW_STEPS:window-steps ACTOR_WARMUP_WINDOWS:warmup-windows \
  CONTROLLER_LR:controller-lr EXPOSURE_TEMPERATURE:exposure-temperature \
  LEARNING_RATE:learning-rate LAMBDA_GT:lambda-gt GAMMA_DISCOUNT:gamma-discount; do
  name="${setting%%:*}"
  option="${setting#*:}"
  if [[ -v "$name" ]]; then
    hyperparameter_args+=("--$option=${!name}")
  fi
done
training_args=(
  --config="$CONFIG"
  --dataset="${DATASET_JSONL:-graph_based_outputs/hypergraph_openthoughts30k_operations_v3.jsonl}"
  --total-updates="$MAX_STEPS" --seed="$TRAIN_SEED" --eval-seed="$EVAL_SEED"
  --teacher-selection=hard --control-batch-size="$DUEL_PROBLEMS"
  --control-pool-size="$HELDOUT_PROBLEMS" --min-valid-controls="$MIN_VALID_CONTROLS"
  --full-opsd-warmup-steps="$WARMUP_STEPS" --mask-teacher-include-final-answer
  --save-every="$SAVE_EVERY" "${hyperparameter_args[@]}" "$@"
)
setup_args=(--run-root="${RUN_ROOT:-$HOME/dsd/outputs}")
if [[ -n "${RUN_DIR:-}" ]]; then
  setup_args+=(--run-dir="$RUN_DIR")
fi
if ! resolved_run_dir="$("${PYTHON_BIN:-python}" -m adaptive_graph_ftrl_redesign.experiment_config "${setup_args[@]}" "${training_args[@]}")"; then
  exit 2
fi
export RUN_DIR="$resolved_run_dir"
export CONFIG="$RUN_DIR/config.yaml"

# Keep socket/temp paths short even when the descriptive output name is long.
export EVAL_WORK_DIR="${EVAL_WORK_DIR:-$(dirname "$RUN_DIR")/.runtime/run-$$-$RANDOM}"
mkdir -p "$RUN_DIR" "$EVAL_WORK_DIR"
EVAL_WORK_DIR="$(cd "$EVAL_WORK_DIR" && pwd)"
export TMPDIR="$EVAL_WORK_DIR/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$EVAL_WORK_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$EVAL_WORK_DIR/torchinductor"
export VLLM_CACHE_ROOT="$EVAL_WORK_DIR/vllm"
export UV_CACHE_DIR="$EVAL_WORK_DIR/uv"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$UV_CACHE_DIR"

printf 'Run: %s\nDuel scoring: %s/%s heldout | FULL warmup: %s | train steps: %s\nEval: checkpoints %s | datasets %s | GPUs %s | train/eval seed %s/%s\n' \
  "$RUN_DIR" "$DUEL_PROBLEMS" "$HELDOUT_PROBLEMS" "$WARMUP_STEPS" "$MAX_STEPS" \
  "$EVAL_STEPS" "$EVAL_DATASETS" "$CUDA_VISIBLE_DEVICES" "$TRAIN_SEED" "$EVAL_SEED"
bash adaptive_graph_ftrl_redesign/scripts/run_train_then_eval.sh \
  "${training_args[@]}" --config="$CONFIG" \
  2>&1 | tee "$RUN_DIR/train_eval.log"
