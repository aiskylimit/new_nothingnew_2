#!/usr/bin/env bash
# IWC driver -- DeepSeek-R1-Distill-Qwen-1.5B, FULL fine-tuning, set up like the SFT Long CoT driver
# (project_commands_r1-qwen-1.5b.sh):
#   data -> capture (spectral + entropy) -> IWC weights -> full-FT IWC SFT -> eval -> compare.
# Training hyperparameters, data format and eval are identical to vanilla-r1-qwen-1.5b; the only
# difference is which response tokens are supervised (spectral selection) and how they are
# weighted (IWC-Stable), so iwc-stable-r1-qwen-1.5b compares directly against vanilla-r1-qwen-1.5b.
# Every phase runs in ONE env: spectral_guided_learning (../spectral_guided_learning.txt).
#   GPUS=0 bash project_commands_iwc_r1-qwen-1.5b.sh
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) each phase runs on (space-separated ids). Training runs torchrun over the whole
# list (effective batch fixed at 32, so 1/2/4/8 GPUs); eval uses the whole list too.
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

# Thinking mode for the SUPERVISION format -- must match the SFT driver so both arms train on the
# same token sequences (long CoT inside R1-Distill's native <think> block).
export ENABLE_THINKING="${ENABLE_THINKING:-true}"

# ============================ TRAIN ============================
# data + capture are shared with the SFT/spectral drivers and skipped when already on disk. The
# parquet must carry step_entropies -- build_iwc_datasets.py stops with a clear error if not.
[[ -f data/r1-qwen-1.5b/train-segmented.jsonl ]]     || bash scripts/data/data_r1-qwen-1.5b.sh
[[ -f data/r1-qwen-1.5b/spectral-strengths.parquet ]] || bash scripts/capture/capture_r1-qwen-1.5b.sh

# IWC-Stable weights (Eq. 13-16), same gentle setting as the qwen3-8b track (project_commands2.sh):
# lambda 0.5, tau 2.0, clip 1.0. Token mass is preserved per sample -- the build log must print
# mean weighted/selected mass=1.000000 for iwc-stable.
export IWC_INTERPOLATION="${IWC_INTERPOLATION:-0.5}"
export IWC_TEMPERATURE="${IWC_TEMPERATURE:-2.0}"
export IWC_CLIP="${IWC_CLIP:-1.0}"
bash scripts/masks/iwc_r1-qwen-1.5b.sh

bash scripts/iwc/train_iwc_r1-qwen-1.5b.sh iwc-stable
# bash scripts/iwc/train_iwc_r1-qwen-1.5b.sh iwc   # Eq. 11 baseline: not mass-preserving

# ============================ EVAL =============================
# Start from a clean shell env so the eval script activates the vLLM env. Thinking is OFF at eval
# regardless of how the data was built -- override with ENABLE_THINKING_EVAL=true.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
ENABLE_THINKING="${ENABLE_THINKING_EVAL:-false}" \
  bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/iwc-stable-r1-qwen-1.5b iwc-stable-r1-qwen-1.5b
# ENABLE_THINKING="${ENABLE_THINKING_EVAL:-false}" \
#   bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/iwc-r1-qwen-1.5b iwc-r1-qwen-1.5b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" "${BASE}/src/compare_results.py"
