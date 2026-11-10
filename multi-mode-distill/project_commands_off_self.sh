#!/usr/bin/env bash
set -euo pipefail

export BASE_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_PATH"

export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

export CKPT="${CKPT:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
export TEACHER_CKPT="${TEACHER_CKPT:-$ASSET_ROOT/models/Qwen2.5_14B-Instruct}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$ASSET_ROOT/processed_data/ultraInteract-v2}"
export DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")}"
RESULTS_ROOT="${RESULTS_ROOT:-$BASE_PATH/results/qwen2.5-1.5B-Instruct-v2}"

export CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
export DEV_NUM="${DEV_NUM:-512}"
export SEED="${SEED:-10}"
export CONTEXT_MAX_NEW_TOKENS="${CONTEXT_MAX_NEW_TOKENS:-${SELF_DISTILL_CONTEXT_MAX_TOKENS:-512}}"
export T_MAX_PROMPT_LENGTH="${T_MAX_PROMPT_LENGTH:-$((MAX_PROMPT_LENGTH + CONTEXT_MAX_NEW_TOKENS))}"
# Reserve the student sequence plus only the extra SELF context budget.
export T_MAX_LENGTH="${T_MAX_LENGTH:-$((MAX_LENGTH + T_MAX_PROMPT_LENGTH - MAX_PROMPT_LENGTH))}"

export RHO_SELF_INIT="${RHO_SELF_INIT:-0.10}"
export RHO_SELF_MAX="${RHO_SELF_MAX:-0.25}"
export RHO_SELF_INCREMENT="${RHO_SELF_INCREMENT:-0.025}"
export ADAPTIVE_THRESHOLD="${ADAPTIVE_THRESHOLD:-0.05}"

export KD_LOSS="${KD_LOSS:-sfkl}"
export KD_RATIO="${KD_RATIO:-0.5}"
export GEOMETRY="${GEOMETRY:-1}"
export CKA="${CKA:-0}"
export MAG_WEIGHT="${MAG_WEIGHT:-2.0}"
export GRAM_WEIGHT="${GRAM_WEIGHT:-10.0}"

RUN_NAME="${RUN_NAME:-off_self_adaptive_${KD_LOSS}_rho${RHO_SELF_INIT}-${RHO_SELF_MAX}_geometry${GEOMETRY}_seed${SEED}}"
export SAVE_PATH="${SAVE_PATH:-$RESULTS_ROOT/$RUN_NAME}"

if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    exit 1
fi

CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT

printf '\n[off-self 1/2] Train Qwen with adaptive OFF+SELF routing (rho_self=%s..%s, increment=%s)\n' \
    "$RHO_SELF_INIT" "$RHO_SELF_MAX" "$RHO_SELF_INCREMENT"
: > "$CHECKPOINT_FILE"
FINETUNE_ENTRYPOINT=finetune_off_self.py \
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash scripts/qwen/train_v2_qwen2.5_14b_to_1.5b.sh "$@"

LORA_PATH="$(<"$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || {
    printf 'Final OFF+SELF LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
}

printf '\n[off-self 2/2] Evaluate checkpoint: %s\n' "$LORA_PATH"
CUDA_DEVICES="$CUDA_DEVICES" LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash scripts/eval/eval.sh run
