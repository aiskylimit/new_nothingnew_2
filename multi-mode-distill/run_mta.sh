#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$PROJECT_ROOT"
PROJECT_ROOT="$PWD"

DISTILLM_FDD_ROOT="$PROJECT_ROOT/distillm-fdd"
ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$PROJECT_ROOT:$DISTILLM_FDD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
RAW_DATA="${RAW_DATA:-$ASSET_ROOT/data/raw/Qwen/Qwen2.5-14B-Instruct/generated_train.jsonl}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract}"
DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/Qwen/Qwen2.5-14B-Instruct}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
DEV_NUM="${DEV_NUM:-200}"
SEED="${SEED:-10}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2}"
LORA_R="${LORA_R:-16}"

RUN_NAME="${RUN_NAME:-adaptive-srkl_bs${BATCH_SIZE}_ga${GRAD_ACC}_lr${LR}_seed${SEED}}"
SAVE_PATH="${SAVE_PATH:-$DISTILLM_FDD_ROOT/results/qwen2.5-1.5B-Instruct-spandistillm/$RUN_NAME}"

if [[ "${PROCESS_DATA:-0}" == 1 ]]; then
    printf '\n[process] UltraInteract with Qwen2.5-14B tokenizer\n'
    "$VENV_PATH/bin/python" "$PROJECT_ROOT/tools/process_data_ultraInteract.py" \
        --base-path "$PROJECT_ROOT" \
        --data-dir "$RAW_DATA" \
        --processed-data-dir "$PROCESSED_DATA_ROOT" \
        --model-path "$TEACHER_CKPT" \
        --model-type qwen \
        --data-process-workers "${DATA_PROCESS_WORKERS:-32}" \
        --max-length "$MAX_LENGTH" \
        --max-prompt-length "$MAX_PROMPT_LENGTH" \
        --dev-num "$DEV_NUM" \
        --seed "$SEED"

    DATA_DIR="$PROCESSED_DATA_ROOT/models/$(basename -- "$TEACHER_CKPT")"
fi

printf '\n[1/2] Train Qwen2.5 span distillation\n'
printf 'Student: %s\nTeacher: %s\nData: %s\nSave: %s\n' \
    "$CKPT" "$TEACHER_CKPT" "$DATA_DIR" "$SAVE_PATH"

CUDA_DEVICES="$CUDA_DEVICES" \
    BASE_PATH="$DISTILLM_FDD_ROOT" ASSET_ROOT="$ASSET_ROOT" \
    CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" DATA_DIR="$DATA_DIR" \
    SAVE_PATH="$SAVE_PATH" MAX_LENGTH="$MAX_LENGTH" \
    MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" DEV_NUM="$DEV_NUM" SEED="$SEED" \
    BATCH_SIZE="$BATCH_SIZE" GRAD_ACC="$GRAD_ACC" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" LR="$LR" EPOCHS="$EPOCHS" \
    LORA_R="$LORA_R" \
    bash "$DISTILLM_FDD_ROOT/scripts/qwen/spandistillm/train_1.5B_14B.sh" "$@"

if [[ "${DRY_RUN:-0}" == 1 || "${SKIP_EVAL:-0}" == 1 ]]; then
    exit 0
fi

FINAL_STEP="$("$VENV_PATH/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["total_iters"])' \
    "$SAVE_PATH/args.json")"
LORA_PATH="$SAVE_PATH/$FINAL_STEP"

if [[ ! -f "$LORA_PATH/adapter_config.json" ]]; then
    printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
fi

printf '\n[2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES="$CUDA_DEVICES" MODEL_PATH="$CKPT" LORA_PATH="$LORA_PATH" \
    SAVE_PATH="$SAVE_PATH" ASSET_ROOT="$ASSET_ROOT" \
    EVAL_DATA_DIR="$EVAL_DATA_DIR" EVAL_MAX_LORA_RANK="$LORA_R" \
    bash "$DISTILLM_FDD_ROOT/scripts/qwen/spandistillm/eval_1.5B_14B.sh" run
