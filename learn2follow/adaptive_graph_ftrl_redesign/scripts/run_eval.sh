#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' \
    "Usage: bash $0 [RUN_DIR] (or set RUN_DIR in the environment)" \
    'Environment options (defaults):' \
    '  EVAL_DATASET=aime25          aime24, aime25, aime26 or hmmt25' \
    '  EVAL_STEPS=25,50             comma-separated checkpoint steps' \
    '  CUDA_VISIBLE_DEVICES=0,1,2,3 four distinct GPUs; TP=4' \
    '  EVAL_SEED=42 VAL_N=12 TEMPERATURE=1.0 TOP_P=0.95 TOP_K=-1' \
    '  MIN_P=0.0 PRESENCE_PENALTY=0.0' \
    '  MAX_NEW_TOKENS=38912 MAX_MODEL_LEN=40960 ENABLE_THINKING=1' \
    '  EVAL_GPU_MEMORY_UTILIZATION=0.9 MODEL_PATH=Qwen/Qwen3-4B' \
    '  EVAL_WORK_DIR=RUN_DIR/eval_runtime RESULTS_DIR=RUN_DIR/eval_DATASET' \
    '  EVAL_MIN_FREE_GIB=10 EVAL_MIN_FREE_INODES=10000 EVAL_PREFLIGHT_ONLY=0' \
    '  UV_PROJECT_ENVIRONMENT=dsd'
  exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if (( $# > 1 )); then
  echo "Usage: bash $0 RUN_DIR (or set RUN_DIR in the environment)" >&2
  exit 2
fi
RUN_DIR="${1:-${RUN_DIR:-}}"
: "${RUN_DIR:?Provide the run directory containing the requested checkpoints}"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
command -v uv >/dev/null || { echo "Install uv as required by graph_based evaluation." >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-dsd}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export TOP_K="${TOP_K:--1}"
export MIN_P="${MIN_P:-0.0}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
export VAL_N="${VAL_N:-12}"
export EVAL_SEED="${EVAL_SEED:-42}"
IFS=',' read -r -a eval_gpus <<< "$CUDA_VISIBLE_DEVICES"
export TENSOR_PARALLEL_SIZE="${#eval_gpus[@]}"
EVAL_STEPS="${EVAL_STEPS:-25,50}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_DATASET="${EVAL_DATASET:-aime25}"
case "$EVAL_DATASET" in
  aime24|aime25|aime26|hmmt25) ;;
  *) echo "EVAL_DATASET must be aime24, aime25, aime26 or hmmt25" >&2; exit 2 ;;
esac
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
case "${ENABLE_THINKING:-1}" in
  1) THINKING_FLAG=--enable_thinking ;;
  0) THINKING_FLAG=--no_thinking ;;
  *) echo "ENABLE_THINKING must be 0 or 1" >&2; exit 2 ;;
esac

# GCC and Python must receive TMPDIR before uv/vLLM start, not after imports.
EVAL_WORK_DIR="${EVAL_WORK_DIR:-/tmp/l2f-${UID}-$$-$RANDOM}"
mkdir -p "$EVAL_WORK_DIR"
EVAL_WORK_DIR="$(cd "$EVAL_WORK_DIR" && pwd)"
export TMPDIR="$EVAL_WORK_DIR/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TRITON_CACHE_DIR="$EVAL_WORK_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$EVAL_WORK_DIR/torchinductor"
export VLLM_CACHE_ROOT="$EVAL_WORK_DIR/vllm"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"
RESULTS_DIR="${RESULTS_DIR:-$RUN_DIR/eval_$EVAL_DATASET}"
mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

preflight() {
  uv run python -m adaptive_graph_ftrl_redesign.eval_preflight \
    --run-dir "$RUN_DIR" --steps "$EVAL_STEPS" --results-dir "$RESULTS_DIR" \
    --min-free-gib "${EVAL_MIN_FREE_GIB:-10}" --min-free-inodes "${EVAL_MIN_FREE_INODES:-10000}"
}

preflight
if [[ "${EVAL_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi
IFS=',' read -r -a STEPS <<< "$EVAL_STEPS"
for step in "${STEPS[@]}"; do
  preflight
  echo "=== ${EVAL_DATASET} checkpoint-${step} | GPUs=${CUDA_VISIBLE_DEVICES} | seed=${EVAL_SEED} | Avg@${VAL_N} ==="
  CHECKPOINT_DIR="$RUN_DIR/checkpoint-${step}" DATASET="$EVAL_DATASET" \
    bash graph_based_scripts/run_graph_based_eval_opsd_4gpu.sh \
      --gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
      --max_model_len "$MAX_MODEL_LEN" "$THINKING_FLAG" \
      --output_file "$RESULTS_DIR/checkpoint-${step}_${EVAL_DATASET}.json" \
      2>&1 | tee "$RESULTS_DIR/checkpoint-${step}_${EVAL_DATASET}.log"
done
