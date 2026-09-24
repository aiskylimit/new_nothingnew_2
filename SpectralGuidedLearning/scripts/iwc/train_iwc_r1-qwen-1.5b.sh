#!/usr/bin/env bash
# Phase 5: train an IWC arm -- DeepSeek-R1-Distill-Qwen-1.5B track.
# Hyperparameters match scripts/spectral/spectral_{full,lora}_r1-qwen-1.5b.sh exactly; the data
# (train-<variant>.jsonl) shares train-spectral's token selection and only adds loss_weights, which
# train_sft.py's MaskedSFTTrainer applies. Never train IWC through train_sft_unsloth.py: it drops them.
# Usage: scripts/iwc/train_iwc_r1-qwen-1.5b.sh {full|lora} [iwc-stable|iwc]
set -euo pipefail

MODE="${1:?mode is required: full or lora}"
VARIANT="${2:-iwc-stable}"
case "${MODE}" in
  full|lora) ;;
  *) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac
case "${VARIANT}" in
  iwc|iwc-stable) ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

read -ra GPUS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DS_SKIP_CUDA_CHECK=1

for _v in $(compgen -e PET_) $(compgen -e TORCHELASTIC_); do unset "$_v"; done
MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --rdzv_backend static \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || "${BASE_PATH}/scripts/setup.sh"
  source "${PROJECT_ENV}/bin/activate"
fi
export PYTHONPATH="${BASE_PATH}/src"
mkdir -p "${BASE_PATH}/logs"

EFFECTIVE_BATCH=32
if (( EFFECTIVE_BATCH % GPUS_PER_NODE != 0 )); then
  echo "train_iwc_r1-qwen-1.5b.sh: ${GPUS_PER_NODE} GPUs does not divide effective batch ${EFFECTIVE_BATCH}." >&2
  exit 2
fi

TAG="${VARIANT}-${MODE}-r1-qwen-1.5b"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_NAME="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH="${BASE_PATH}/data/r1-qwen-1.5b/train-${VARIANT}.jsonl"
OUTPUT_DIR="${BASE_PATH}/checkpoints/${TAG}"
EPOCHS=3
LR="${LR:-5.0e-5}"
MIN_LR="${MIN_LR:-1.0e-5}"
WARMUP_RATIO=0.1
BATCH_SIZE=1
GRAD_ACC=$(( EFFECTIVE_BATCH / GPUS_PER_NODE ))
ATTN=sdpa
LOG_INTERVAL=5
SEED=42
SAVE_STRATEGY=epoch
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=6
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
DS_CONFIG="${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"
MAX_SEQ_LEN=32768

[[ -f "${DATA_PATH}" ]] || { echo "missing ${DATA_PATH} -- run scripts/masks/iwc_r1-qwen-1.5b.sh first" >&2; exit 1; }

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
OPTS+=" --attn-implementation ${ATTN}"
OPTS+=" --logging-steps ${LOG_INTERVAL}"
OPTS+=" --save-strategy ${SAVE_STRATEGY}"
OPTS+=" --save-steps ${SAVE_STEPS}"
OPTS+=" --save-total-limit ${SAVE_TOTAL_LIMIT}"
OPTS+=" --seed ${SEED}"
if [[ "${MODE}" == lora ]]; then
  OPTS+=" --use-lora"
  OPTS+=" --lora-r ${LORA_R}"
  OPTS+=" --lora-alpha ${LORA_ALPHA}"
  OPTS+=" --lora-dropout ${LORA_DROPOUT}"
  OPTS+=" --lora-target-modules ${LORA_TARGET_MODULES}"
  OPTS+=" --no-lora-merge"
else
  OPTS+=" --no-use-lora"
fi
OPTS+=" --deepspeed-config ${DS_CONFIG}"
OPTS+=" --max-seq-len ${MAX_SEQ_LEN}"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/train_sft.py ${OPTS}"
echo "${CMD}"
${CMD} 2>&1 | tee "${BASE_PATH}/logs/${TAG}.log"
