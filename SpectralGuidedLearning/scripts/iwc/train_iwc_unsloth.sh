#!/usr/bin/env bash
set -euo pipefail

TRACK="${1:?track is required: qwen25-7b or qwen3-8b}"
VARIANT="${2:?variant is required: iwc or iwc-stable}"
case "${TRACK}" in
  qwen25-7b) MODEL_DIR="Qwen2.5-7B-Instruct" ;;
  qwen3-8b) MODEL_DIR="Qwen3-8B" ;;
  *) echo "unknown track: ${TRACK}" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  iwc|iwc-stable) ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

read -ra GPUS <<< "${GPUS:-0}"
if [[ "${#GPUS[@]}" -ne 1 ]]; then
  echo "train_iwc_unsloth.sh: single-GPU only. Got GPUS='${GPUS[*]}'." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral-guided-learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/${MODEL_DIR}"
DATA_PATH="${BASE_PATH}/data/${TRACK}/train-${VARIANT}.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${VARIANT}-unsloth-${TRACK}"
OPTIM=adamw_8bit

OPTS="--model-name ${MODEL_NAME} --data-path ${DATA_PATH} --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs 3 --learning-rate 5.0e-5 --min-learning-rate 1.0e-5 --warmup-ratio 0.1"
OPTS+=" --per-device-batch-size 1 --gradient-accumulation-steps 8"
OPTS+=" --logging-steps 5 --save-strategy epoch --save-steps 500 --save-total-limit 6 --seed 42"
OPTS+=" --optim ${OPTIM} --max-seq-len 32768 --no-use-lora"

CMD="python ${BASE_PATH}/src/train_sft_unsloth.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${VARIANT}-unsloth-${TRACK}.log"
