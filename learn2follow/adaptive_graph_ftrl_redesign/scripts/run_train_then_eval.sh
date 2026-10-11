#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
CONFIG="${CONFIG:-adaptive_graph_ftrl_redesign/configs/graph_answer.yaml}"
DATASET_JSONL="${DATASET_JSONL:-graph_based_outputs/hypergraph_openthoughts30k_operations_v3.jsonl}"
MAX_STEPS="${MAX_STEPS:-100}"
VARIANT_NAME="$(basename "$CONFIG" .yaml)"
RUN_DIR="${RUN_DIR:-adaptive_graph_ftrl_redesign/outputs/${VARIANT_NAME}_step${MAX_STEPS}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export GRAPH_TRAIN_LAUNCH_ID
GRAPH_TRAIN_LAUNCH_ID="$("$PYTHON_BIN" -c 'import uuid; print(uuid.uuid4().hex)')"
TRAIN_STATUS=0
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_PROCESSES" \
  -m adaptive_graph_ftrl_redesign.train --config "$CONFIG" --dataset "$DATASET_JSONL" \
  --output-dir "$RUN_DIR" --total-updates "$MAX_STEPS" --seed "${TRAIN_SEED:-42}" "$@" \
  || TRAIN_STATUS=$?
if ! "$PYTHON_BIN" -m adaptive_graph_ftrl_redesign.lifecycle \
  --run-dir "$RUN_DIR" --launch-id "$GRAPH_TRAIN_LAUNCH_ID" --steps "$MAX_STEPS" --world "$NUM_PROCESSES"; then
  echo "Training completion could not be verified; evaluation will not run." >&2
  if (( TRAIN_STATUS != 0 )); then
    exit "$TRAIN_STATUS"
  fi
  exit 1
fi
if (( TRAIN_STATUS != 0 )); then
  echo "WARNING: training launcher exited ${TRAIN_STATUS} after verified completion. Evaluating saved checkpoints; the final exit code will retain the training failure." >&2
fi
"$PYTHON_BIN" -m adaptive_graph_ftrl_redesign.evaluate --run-dir "$RUN_DIR" \
  --steps "${EVAL_STEPS:-25,50,75,100}" --datasets "${EVAL_DATASETS:-aime25}" \
  --gpus "$CUDA_VISIBLE_DEVICES" --seed "${EVAL_SEED:-42}" \
  --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
exit "$TRAIN_STATUS"
