#!/usr/bin/env bash
set -euo pipefail

# Train Gemma on already-processed data, then evaluate the final LoRA checkpoint.
BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"

export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
DATA_ROOT="${DATA_ROOT:-$ASSET_ROOT}"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    printf 'Training virtualenv not found: %s\n' "$VENV_PATH" >&2
    exit 1
fi
source "$VENV_PATH/bin/activate"

export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-$DATA_ROOT/models/google_gemma-2-2b-it}"
TEACHER_CKPT="${TEACHER_CKPT:-$DATA_ROOT/models/google_gemma-2-9b-it}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$DATA_ROOT/processed_data/ultraInteract-v2}"
DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ASSET_ROOT/data/eval}"
GEMMA_RESULTS_ROOT="${GEMMA_RESULTS_ROOT:-$BASE_PATH/results/gemma-2-2b-it-distill}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
DEV_NUM="${DEV_NUM:-512}"
SEED="${SEED:-10}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
RUN_EVAL="${RUN_EVAL:-1}"

KD_LOSS="${KD_LOSS:-sfkl}"
KD_RATIO="${KD_RATIO:-0.5}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
FINETUNE_ENTRYPOINT="${FINETUNE_ENTRYPOINT:-finetune.py}"
EXCLUDE_OFF_POLICY="${EXCLUDE_OFF_POLICY:-0}"

GEOMETRY="${GEOMETRY:-0}"
CKA="${CKA:-0}"
MAG_WEIGHT="${MAG_WEIGHT:-0.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-0.0}"
CKA_WEIGHT="${CKA_WEIGHT:-0.0}"
MENGER_WEIGHT="${MENGER_WEIGHT:-0.0}"
MENGER_EPS="${MENGER_EPS:-1.0e-6}"

case "$RUN_EVAL" in
    0|1) ;;
    *) printf 'RUN_EVAL must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$GEOMETRY" in
    0|1) ;;
    *) printf 'GEOMETRY must be 0 or 1\n' >&2; exit 2 ;;
esac
case "$CKA" in
    0|1) ;;
    *) printf 'CKA must be 0 or 1\n' >&2; exit 2 ;;
esac
if [[ "$GEOMETRY" == 1 && "$CKA" == 1 ]]; then
    printf 'GEOMETRY and CKA are mutually exclusive\n' >&2
    exit 2
fi

for required_path in "$CKPT" "$TEACHER_CKPT"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'Required model path not found: %s\n' "$required_path" >&2
        exit 1
    fi
done
if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-${FINETUNE_ENTRYPOINT%.py}_geo${GEOMETRY}_cka${CKA}_menger${MENGER_WEIGHT}}"
SAVE_PATH="${SAVE_PATH:-$GEMMA_RESULTS_ROOT/$RUN_NAME}"
CHECKPOINT_FILE="$(mktemp)"
trap 'rm -f -- "$CHECKPOINT_FILE"' EXIT

printf '\n[%s 1/2] Train Gemma: KD=%s, geometry=%s, CKA=%s, Menger=%s\n' \
    "$RUN_NAME" "$KD_RATIO" "$GEOMETRY" "$CKA" "$MENGER_WEIGHT"
: > "$CHECKPOINT_FILE"
CUDA_DEVICES="$CUDA_DEVICES" DATA_DIR="$DATA_DIR" \
    BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" VENV_PATH="$VENV_PATH" \
    CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" \
    PROCESSED_DATA_ROOT="$PROCESSED_DATA_ROOT" SAVE_PATH="$SAVE_PATH" \
    MAX_LENGTH="$MAX_LENGTH" MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
    DEV_NUM="$DEV_NUM" SEED="$SEED" \
    KD_LOSS="$KD_LOSS" KD_RATIO="$KD_RATIO" SKEW_ALPHA="$SKEW_ALPHA" \
    GEOMETRY="$GEOMETRY" CKA="$CKA" \
    MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" \
    CKA_WEIGHT="$CKA_WEIGHT" MENGER_WEIGHT="$MENGER_WEIGHT" \
    MENGER_EPS="$MENGER_EPS" DISTILL_TOP_K="$DISTILL_TOP_K" \
    DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    FINETUNE_ENTRYPOINT="$FINETUNE_ENTRYPOINT" \
    EXCLUDE_OFF_POLICY="$EXCLUDE_OFF_POLICY" \
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash "$BASE_PATH/scripts/gemma/train_gemma2_9b_to_2b.sh" "$@"

if [[ ! -s "$CHECKPOINT_FILE" ]]; then
    printf 'Training finished without writing FINAL_CHECKPOINT_FILE: %s\n' "$CHECKPOINT_FILE" >&2
    exit 1
fi
LORA_PATH="$(<"$CHECKPOINT_FILE")"
if [[ ! -f "$LORA_PATH/adapter_config.json" ]]; then
    printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
fi

if [[ "$RUN_EVAL" == 0 ]]; then
    printf '\n[%s 2/2] Evaluation skipped (RUN_EVAL=0)\n' "$RUN_NAME"
    exit 0
fi

printf '\n[%s 2/2] Evaluate checkpoint: %s\n' "$RUN_NAME" "$LORA_PATH"
CUDA_DEVICES="$CUDA_DEVICES" LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" \
    EVAL_DATA_DIR="$EVAL_DATA_DIR" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash "$BASE_PATH/scripts/eval/eval_gemma.sh" run
