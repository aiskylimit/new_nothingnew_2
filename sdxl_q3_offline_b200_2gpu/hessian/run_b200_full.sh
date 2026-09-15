#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT
source "$PROJECT_ROOT/env.sh"
source "$RUNTIME_ROOT/b200-autotune.env"

export PIPELINE_MODE=full851k
export DATASET_PAIRS=851293
export OFFLINE_EVAL_LIMIT=0
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
export CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-2}"
export RUN_NAME="${RUN_NAME:-q3_dspo_sdxl_full851k_b200x2_mb${TRAIN_BATCH_SIZE}_eb${EFFECTIVE_BATCH}}"

bash "$PROJECT_ROOT/project_command.sh"
