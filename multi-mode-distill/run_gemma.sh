#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"

export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"

DATA_ROOT="${DATA_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
VENV_PATH="${VENV_PATH:-/mnt/local/uvenvs/reasoning-velocity-distill}"
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$BASE_PATH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Training data and models are local to this project; eval data is shared.
RAW_DATA="${RAW_DATA:-$DATA_ROOT/data/raw/google/gemma-2-9b-it/generated_train.jsonl}"
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

# uv venv --python 3.11 "$VENV_PATH"
# source "$VENV_PATH/bin/activate"
# uv pip install -r "$BASE_PATH/multi-mode-distill.txt"

# mkdir -p -- "$(dirname -- "$RAW_DATA")" "$CKPT" "$TEACHER_CKPT"
# hf download VoCuc/UltraInteract-Infer \
#     google/gemma-2-9b-it/generated_train.jsonl \
#     --repo-type dataset \
#     --local-dir "$DATA_ROOT/data/raw"
# hf download google/gemma-2-2b-it --local-dir "$CKPT"
# hf download google/gemma-2-9b-it --local-dir "$TEACHER_CKPT"

# mkdir -p -- "$EVAL_DATA_DIR/code_eval"
# hf download openai/gsm8k --repo-type dataset --local-dir "$EVAL_DATA_DIR/gsm8k"
# hf download qintongli/GSM-Plus --repo-type dataset --local-dir "$EVAL_DATA_DIR/gsm_plus"
# hf download EleutherAI/hendrycks_math --repo-type dataset --local-dir "$EVAL_DATA_DIR/hendrycks_math"
# hf download google-research-datasets/mbpp --repo-type dataset --local-dir "$EVAL_DATA_DIR/mbpp"
# hf download allenai/sciq --repo-type dataset --local-dir "$EVAL_DATA_DIR/sciq"
# hf download cais/mmlu --repo-type dataset --local-dir "$EVAL_DATA_DIR/mmlu"
# hf download TIGER-Lab/MMLU-Pro --repo-type dataset --local-dir "$EVAL_DATA_DIR/mmlu_pro"
# hf download SaylorTwift/bbh --repo-type dataset --local-dir "$EVAL_DATA_DIR/bbh"
# curl --fail --location \
#     https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/code_eval.py \
#     --output "$EVAL_DATA_DIR/code_eval/code_eval.py"
# curl --fail --location \
#     https://raw.githubusercontent.com/huggingface/evaluate/v0.4.6/metrics/code_eval/execute.py \
#     --output "$EVAL_DATA_DIR/code_eval/execute.py"

"$VENV_PATH/bin/python" "$BASE_PATH/tools/process_data_ultraInteract.py" \
    --base-path "$BASE_PATH" \
    --data-dir "$RAW_DATA" \
    --processed-data-dir "$PROCESSED_DATA_ROOT" \
    --model-path "$CKPT" \
    --model-type gemma \
    --data-process-workers "${DATA_PROCESS_WORKERS:-32}" \
    --max-length "$MAX_LENGTH" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --dev-num "$DEV_NUM" \
    --seed "$SEED"

KD_LOSS="${KD_LOSS:-sfkl}"
KD_RATIO="${KD_RATIO:-0.5}"
SKEW_ALPHA="${SKEW_ALPHA:-0.1}"
CKA="${CKA:-0}"
DEFAULT_GEOMETRY=1
if [[ "$CKA" == 1 ]]; then DEFAULT_GEOMETRY=0; fi
GEOMETRY="${GEOMETRY:-$DEFAULT_GEOMETRY}"
MAG_WEIGHT="${MAG_WEIGHT:-2.0}"
GRAM_WEIGHT="${GRAM_WEIGHT:-10.0}"
CKA_WEIGHT="${CKA_WEIGHT:-1.0}"
MENGER_WEIGHT="${MENGER_WEIGHT:-0.0}"
MENGER_EPS="${MENGER_EPS:-1.0e-6}"
DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
RUN_NAME="${RUN_NAME:-geo${GEOMETRY}_cka${CKA}_menger${MENGER_WEIGHT}}"
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
    FINAL_CHECKPOINT_FILE="$CHECKPOINT_FILE" \
    bash "$BASE_PATH/scripts/gemma/train_gemma2_9b_to_2b.sh" "$@"

LORA_PATH="$(<"$CHECKPOINT_FILE")"
[[ -f "$LORA_PATH/adapter_config.json" ]] || {
    printf 'Final LoRA checkpoint missing: %s\n' "$LORA_PATH" >&2
    exit 1
}

printf '\n[%s 2/2] Evaluate checkpoint: %s\n' "$RUN_NAME" "$LORA_PATH"
CUDA_DEVICES="$CUDA_DEVICES" LORA_PATH="$LORA_PATH" MODEL_PATH="$CKPT" \
    BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" \
    EVAL_DATA_DIR="$EVAL_DATA_DIR" \
    SAVE_PATH="$(dirname -- "$LORA_PATH")" \
    EVAL_MAX_LORA_RANK="${EVAL_MAX_LORA_RANK:-${LORA_R:-16}}" \
    bash "$BASE_PATH/scripts/eval/eval_gemma.sh" run
