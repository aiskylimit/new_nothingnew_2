#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0,1}}}"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE=${#GPUS[@]}
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NNODES"
    --node_rank "${NODE_RANK:-0}"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
)

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"
ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"

CKPT_NAME="qwen2.5-1.5B-Instruct"
TEACHER_CKPT_NAME="qwen2.5-14B-Instruct"
CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
DATA_DIR="${DATA_DIR:-$ASSET_ROOT/processed_data/ultraInteract/Qwen/Qwen2.5-14B-Instruct}"
DATA_DIR="${DATA_DIR%/}/"
DS_CONFIG="${DS_CONFIG:-$BASE_PATH/configs/deepspeed/ds_config_bf16.json}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
DEV_NUM="${DEV_NUM:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
KD_RATIO="${KD_RATIO:-1.0}"
W_SPAN_LOSS="${W_SPAN_LOSS:-2.0}"
SEED="${SEED:-10}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

SAVE_PATH="${SAVE_PATH:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-spandistillm/adaptive-srkl_bs${BATCH_SIZE}_ga${GRAD_ACC}_lr${LR}_seed${SEED}}"

OPTS=(
    --base-path "$BASE_PATH"
    --model-path "$CKPT" --model-type qwen --ckpt-name "$CKPT_NAME"
    --teacher-model-path "$TEACHER_CKPT" --teacher-model-type qwen
    --teacher-ckpt-name "$TEACHER_CKPT_NAME" --teacher-model-fp16
    --n-gpu "$GPUS_PER_NODE" --n-nodes "$NNODES" --bf16
    --data-dir "$DATA_DIR" --num-workers "$NUM_WORKERS" --dev-num "$DEV_NUM"
    --lr "$LR" --batch-size "$BATCH_SIZE" --eval-batch-size "$EVAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACC" --gradient-checkpointing
    --lr-decay-style cosine --warmup-iters 0
    --weight-decay 1e-2 --clip-grad 1.0 --epochs "$EPOCHS"
    --kd-ratio "$KD_RATIO" --w-span-loss "$W_SPAN_LOSS"
    --max-length "$MAX_LENGTH" --max-prompt-length "$MAX_PROMPT_LENGTH"
    --do-train --do-valid
    --save-interval -1 --eval-interval "$EVAL_INTERVAL"
    --log-interval 10 --mid-log-num 0 --save "$SAVE_PATH"
    --seed "$SEED"
    --deepspeed --deepspeed_config "$DS_CONFIG"
    --type adaptive-srkl --student-gen
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0
    --repetition-penalty 1.0 --gen-num-beams 1 --gen-top-p 1.0
    --init-threshold 0.0 --loss-eps 0.1 --capacity 1000
    --teacher_layer_mapping 24 36 48
    --student_layer_mapping 14 21 28
    --split_layer_mapping 0 1 3 3
    --peft lora --peft-lora-r "$LORA_R"
    --peft-lora-alpha "$LORA_ALPHA" --peft-lora-dropout "$LORA_DROPOUT"
)

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CODE_BASE=HF

CMD=(torchrun "${DISTRIBUTED_ARGS[@]}" "$BASE_PATH/span_finetune.py" "${OPTS[@]}" "$@")
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

mkdir -p -- "$SAVE_PATH"
exec "${CMD[@]}"
