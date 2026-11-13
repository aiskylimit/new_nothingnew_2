#!/usr/bin/env bash
# Inspect GPT-2 layer correspondence on one GPU; no training or evaluation.
set -euo pipefail

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "Invalid RUN_ID" >&2
    exit 2
fi
OUTPUT_ROOT="${BASE_PATH}/results/gpt2/cka_runs/${RUN_ID}"
LOG_DIR="${BASE_PATH}/logs/gpt2_cka/${RUN_ID}"
mkdir -p "$(dirname "${OUTPUT_ROOT}")"
if [ -e "${OUTPUT_ROOT}" ]; then
    echo "CKA output already exists: ${OUTPUT_ROOT}" >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH="${BASE_PATH}"
echo "CKA on GPU 6. Report: ${OUTPUT_ROOT}/report.md; log: ${LOG_DIR}/cka.log"
exec python "${BASE_PATH}/tools/cka_layer_pairing.py" \
    --teacher "${TEACHER_PATH}" \
    --student "${STUDENT_PATH}" \
    --data "${RAW_DATA_ROOT}/dolly/raw.jsonl" \
    --output "${OUTPUT_ROOT}" \
    > "${LOG_DIR}/cka.log" 2>&1
