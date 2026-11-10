#!/usr/bin/env bash
set -euo pipefail

# Run Gemma ablations. The ON+SELF variant excludes OFF-policy completely.
BASE_PATH="${BASE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_PATH"
BASE_PATH="$PWD"

export ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"
DATA_ROOT="${DATA_ROOT:-$ASSET_ROOT}"
CKPT="${CKPT:-$DATA_ROOT/models/google_gemma-2-2b-it}"
TEACHER_CKPT="${TEACHER_CKPT:-$DATA_ROOT/models/google_gemma-2-9b-it}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$DATA_ROOT/processed_data/ultraInteract-v2}"
DATA_DIR="${DATA_DIR:-$PROCESSED_DATA_ROOT/models/$(basename -- "$CKPT")}"
GEMMA_RESULTS_ROOT="${GEMMA_RESULTS_ROOT:-$BASE_PATH/results/gemma-2-2b-it-distill}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RUN_EVAL="${RUN_EVAL:-1}"
DISTILL_TOP_K="${DISTILL_TOP_K:-5120}"

# Routing ablations compare only the distillation modes. Disable all optional
# representation/geometry objectives so they cannot confound the comparison.
GEOMETRY=0
CKA=0
MENGER_WEIGHT=0
MAG_WEIGHT=0
GRAM_WEIGHT=0
CKA_WEIGHT=0

case "$RUN_EVAL" in
    0|1) ;;
    *) printf 'RUN_EVAL must be 0 or 1\n' >&2; exit 2 ;;
esac

if [[ ! -s "$DATA_DIR/train.jsonl" || ( ! -s "$DATA_DIR/valid.jsonl" && ! -s "$DATA_DIR/dev.jsonl" ) ]]; then
    printf 'Processed train and valid/dev JSONL files are required in: %s\n' "$DATA_DIR" >&2
    printf 'This ablation runner does not process data. Set DATA_DIR to prepared data.\n' >&2
    exit 1
fi

printf '\n[data] Reuse processed Gemma data: %s\n' "$DATA_DIR"
printf '[loss] Geometry=off, CKA=off, Menger=off\n'

run_off_self() {
    local run_name="${OFF_SELF_RUN_NAME:-ablation_off_self_adaptive}"
    printf '\n[ablation:off_self] Adaptive OFF-policy + SELF only\n'
    FINETUNE_ENTRYPOINT=finetune_off_self.py \
        RUN_NAME="$run_name" SAVE_PATH="${OFF_SELF_SAVE_PATH:-$GEMMA_RESULTS_ROOT/$run_name}" \
        BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" DATA_ROOT="$DATA_ROOT" \
        CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" \
        PROCESSED_DATA_ROOT="$PROCESSED_DATA_ROOT" DATA_DIR="$DATA_DIR" \
        GEOMETRY="$GEOMETRY" CKA="$CKA" MENGER_WEIGHT="$MENGER_WEIGHT" \
        MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" CKA_WEIGHT="$CKA_WEIGHT" \
        DISTILL_TOP_K="$DISTILL_TOP_K" \
        CUDA_DEVICES="$CUDA_DEVICES" RUN_EVAL="$RUN_EVAL" \
        bash "$BASE_PATH/run_gemma.sh" "$@"
}

run_on_self() {
    local run_name="${ON_SELF_RUN_NAME:-ablation_on_self_adaptive}"
    printf '\n[ablation:on_self] Adaptive ON + SELF only (no OFF-policy)\n'
    FINETUNE_ENTRYPOINT=finetune.py EXCLUDE_OFF_POLICY=1 \
        RUN_NAME="$run_name" SAVE_PATH="${ON_SELF_SAVE_PATH:-$GEMMA_RESULTS_ROOT/$run_name}" \
        BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" DATA_ROOT="$DATA_ROOT" \
        CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" \
        PROCESSED_DATA_ROOT="$PROCESSED_DATA_ROOT" DATA_DIR="$DATA_DIR" \
        GEOMETRY="$GEOMETRY" CKA="$CKA" MENGER_WEIGHT="$MENGER_WEIGHT" \
        MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" CKA_WEIGHT="$CKA_WEIGHT" \
        DISTILL_TOP_K="$DISTILL_TOP_K" \
        CUDA_DEVICES="$CUDA_DEVICES" RUN_EVAL="$RUN_EVAL" \
        bash "$BASE_PATH/run_gemma.sh" "$@"
}

run_no_adaptive() {
    local off_ratio="${NO_ADAPTIVE_OFF_RATIO:-0.3333333333333333}"
    local self_ratio="${NO_ADAPTIVE_SELF_RATIO:-0.3333333333333333}"
    local on_ratio="${NO_ADAPTIVE_ON_RATIO:-0.3333333333333333}"
    local run_name="${NO_ADAPTIVE_RUN_NAME:-ablation_no_adaptive_off${off_ratio}_self${self_ratio}_on${on_ratio}}"
    printf '\n[ablation:no_adaptive] Fixed OFF=%s SELF=%s ON=%s\n' \
        "$off_ratio" "$self_ratio" "$on_ratio"
    MODEL_TYPE=gemma TEACHER_MODEL_TYPE=gemma \
        CKPT_NAME=gemma-2-2b-it TEACHER_CKPT_NAME=gemma-2-9b-it \
        OFF_POLICY_RATIO="$off_ratio" SELF_DISTILL_RATIO="$self_ratio" \
        ON_POLICY_RATIO="$on_ratio" \
        SAVE_PATH="${NO_ADAPTIVE_SAVE_PATH:-$GEMMA_RESULTS_ROOT/$run_name}" \
        EVAL_SCRIPT="$BASE_PATH/scripts/eval/eval_gemma.sh" \
        BASE_PATH="$BASE_PATH" ASSET_ROOT="$ASSET_ROOT" \
        CKPT="$CKPT" TEACHER_CKPT="$TEACHER_CKPT" \
        PROCESSED_DATA_ROOT="$PROCESSED_DATA_ROOT" DATA_DIR="$DATA_DIR" \
        GEOMETRY="$GEOMETRY" CKA="$CKA" MENGER_WEIGHT="$MENGER_WEIGHT" \
        MAG_WEIGHT="$MAG_WEIGHT" GRAM_WEIGHT="$GRAM_WEIGHT" CKA_WEIGHT="$CKA_WEIGHT" \
        DISTILL_TOP_K="$DISTILL_TOP_K" \
        CUDA_DEVICES="$CUDA_DEVICES" RUN_EVAL="$RUN_EVAL" \
        bash "$BASE_PATH/run_no_adaptive.sh" "$@"
}

# after the current experiment (including evaluation) completes successfully.
run_off_self "$@"
run_on_self "$@"
run_no_adaptive "$@"
