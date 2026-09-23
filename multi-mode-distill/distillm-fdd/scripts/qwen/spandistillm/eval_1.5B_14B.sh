#!/usr/bin/env bash
set -euo pipefail

DISTILLM_FDD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT_ROOT="$(cd -- "$DISTILLM_FDD_ROOT/.." && pwd)"
ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/reasoning_velocity_distill}"

export ASSET_ROOT
export MODEL_PATH="${MODEL_PATH:-$ASSET_ROOT/models/Qwen2.5_1.5B-Instruct}"
export SAVE_PATH="${SAVE_PATH:-$DISTILLM_FDD_ROOT/results/qwen2.5-1.5B-Instruct-spandistillm/adaptive-srkl_bs8_ga4_lr1e-4_seed10}"

exec "$PROJECT_ROOT/scripts/eval/eval.sh" "$@"
