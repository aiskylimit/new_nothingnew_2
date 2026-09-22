#!/usr/bin/env bash
# Phase 5: SFT Long CoT (vanilla arm) -- DeepSeek-R1-Distill-Qwen-1.5B track.
# Every response token supervised (all-ones mask from data_r1-qwen-1.5b.sh), FULL fine-tuning:
# Unsloth on a SINGLE GPU (no LoRA, no DeepSpeed, no torchrun), same trainer/knobs as
# scripts/spectral/spectral_r1-qwen-1.5b.sh so the two arms differ only in the loss mask.
set -euo pipefail

read -ra GPUS <<< "${GPUS:-0 1}"
export CUDA_VISIBLE_DEVICES="${GPUS[0]}"   # Unsloth OSS is single-GPU; pin the first listed GPU
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Unsloth needs its own venv (torch 2.9 / transformers 4.57 -- see spectral_guided_learning_train.txt);
# the main env (torch 2.13 / vllm) cannot hold it, so never fall back to scripts/setup.sh here.
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning_train}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || {
    echo "ERROR: unsloth train env not found at ${PROJECT_ENV}; build it from spectral_guided_learning_train.txt or set PROJECT_ENV" >&2
    exit 1
  }
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH="${BASE_PATH}/data/r1-qwen-1.5b/train-vanilla.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/vanilla-r1-qwen-1.5b"
EPOCHS=3
LR=1.0e-5              # full-FT lr (LoRA would use 5e-5); matched to the spectral arm
MIN_LR=1.0e-6
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=8            # bs1 x ga8 = effective batch 8 (single GPU)
OPTIM=adamw_torch     # 1.5B optimizer state fits easily; 7B uses adamw_8bit
LOG_INTERVAL=5
SEED=42
SAVE_STRATEGY=epoch
SAVE_TOTAL_LIMIT=2
MAX_SEQ_LEN=32768

OPTS=""
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --output-dir ${OUTPUT_DIR}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --learning-rate ${LR}"
OPTS+=" --min-learning-rate ${MIN_LR}"
OPTS+=" --warmup-ratio ${WARMUP_RATIO}"
OPTS+=" --per-device-batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --optim ${OPTIM}"
OPTS+=" --logging-steps ${LOG_INTERVAL}"
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"
OPTS+=" --no-use-lora"

CMD="python ${BASE_PATH}/src/train_sft_unsloth.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/vanilla-r1-qwen-1.5b.log"
