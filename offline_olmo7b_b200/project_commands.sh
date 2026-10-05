#!/bin/bash
# Eval-only: re-evaluate RLSD and SDPO on OLMo-3-7B-Think from checkpoints already on this server.
# Nothing is trained here. Results go to *_reeval directories so existing results are not overwritten.
# Uses GPU 0 (main) + GPU 0,1 (vLLM replicas) - the 2-GPU layout of this B200 server.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH=.

source /mnt/local/uvenvs/tropic/bin/activate

PROJECT_NAME="aiskylimit_new_nothingnew_2"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_OLMO="${BASE_DIR}/models/Olmo-3-7B-Think"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

MAIN_GPU=0
EVAL_VLLM_GPU_IDS="0,1"
VLLM_BASE_PORT=8100
VLLM_EXECUTABLE="vllm"
GPU_MEM_UTIL=0.9
VLLM_MAX_MODEL_LEN=41000
CHECKPOINTS="25 50 75 100"
BENCHMARKS="aime25 aime26 hmmt25"

eval_checkpoint() {
  local script=$1 tag=$2 step=$3
  local ckpt="${ROOT}/results_${tag}_olmo7b/${tag}_checkpoint_step${step}"
  local outdir="${ROOT}/results_${tag}_olmo7b_reeval"
  local log="${outdir}/${tag}_reeval_step${step}.log"
  mkdir -p "$outdir"
  if [ ! -d "$ckpt" ]; then echo "skip (no checkpoint): $ckpt"; return 0; fi
  if [ -f "$log" ] && grep -q "ALL DONE" "$log"; then echo "skip (already done): $log"; return 0; fi
  echo "=== eval: $script step ${step} ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${MODEL_OLMO}" --skip-train --checkpoint-path "$ckpt" \
      --output-dir "$outdir" \
      --eval-engine vllm --vllm-gpu-ids "${EVAL_VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      --eval-benchmarks ${BENCHMARKS} \
      2>&1 | tee "$log"
}

for step in $CHECKPOINTS; do eval_checkpoint run_rlsd_experiment_olmo7b.py rlsd "$step"; done
for step in $CHECKPOINTS; do eval_checkpoint run_sdpo_experiment_olmo7b.py sdpo "$step"; done

echo "ALL DONE - RLSD and SDPO re-eval finished"
