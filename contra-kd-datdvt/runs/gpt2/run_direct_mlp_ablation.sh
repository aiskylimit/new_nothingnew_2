#!/usr/bin/env bash
# Pretrain one direct MLP, then train DistillM and FDD branches in parallel.
set -euo pipefail
set -m

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
BASE_PORT=${BASE_PORT:-20120}
LOG_DIR=${LOG_DIR:-"${BASE_PATH}/logs/gpt2_direct_mlp_ablation"}
mkdir -p "${LOG_DIR}"
export PET_STANDALONE=1
export PYTHONPATH="${BASE_PATH}"

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill -- "-${pid}" 2>/dev/null || true
    done
}
PIDS=()
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Pretraining direct MLP on GPU 6..."
CUDA_VISIBLE_DEVICES=6 bash "${BASE_PATH}/runs/gpt2/direct_mlp.sh" \
    "${BASE_PATH}" "${BASE_PORT}" pretrain \
    > "${LOG_DIR}/pretrain.log" 2>&1

# Both branches consume the same MLP checkpoint but write separate model outputs.
CUDA_VISIBLE_DEVICES=6 EVAL_PORT=$((BASE_PORT + 20)) \
    bash "${BASE_PATH}/runs/gpt2/direct_mlp.sh" \
    "${BASE_PATH}" "$((BASE_PORT + 10))" distillm \
    > "${LOG_DIR}/distillm.log" 2>&1 &
distillm_pid=$!
PIDS+=("${distillm_pid}")

CUDA_VISIBLE_DEVICES=7 EVAL_PORT=$((BASE_PORT + 21)) \
    bash "${BASE_PATH}/runs/gpt2/direct_mlp.sh" \
    "${BASE_PATH}" "$((BASE_PORT + 11))" fdd \
    > "${LOG_DIR}/fdd.log" 2>&1 &
fdd_pid=$!
PIDS+=("${fdd_pid}")

echo "Direct MLP: DistillM on GPU 6 and FDD on GPU 7. Logs: ${LOG_DIR}"
wait "${distillm_pid}"
wait "${fdd_pid}"
PIDS=()
echo "Both direct MLP branches completed."
