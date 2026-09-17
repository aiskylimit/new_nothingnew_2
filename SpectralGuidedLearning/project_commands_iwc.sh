#!/usr/bin/env bash
# IWC experiment driver -- long-CoT data -> spectral capture -> IWC masks -> IWC SFT -> EVAL -> compare.
# Runs alongside project_commands.sh (answer-only arm) on a DIFFERENT GPU: every path below is
# disjoint from that driver (data/<track>/ vs data/<track>-answer/, checkpoints/iwc-*, logs/iwc-*,
# results/iwc-*). Only compare_results.py rewrites the shared results/comparison-table.md and
# results/eval-summary.json, and it regenerates them from every results/<tag>/ each time, so
# whichever driver finishes last simply produces the fuller table.
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) every phase script runs on (space-separated ids). Override: GPUS="0" ./project_commands_iwc.sh
# Must NOT be the GPU project_commands.sh is training on: eval reserves 90% of its GPU for vLLM.
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

# ============================ TRAIN ============================
# per track: data -> capture (spectral + entropy) -> IWC weights/masks -> IWC SFT (unsloth, 1 GPU).
# IWC and IWC-Stable share exactly the spectral-selected token set; only weights differ.
# Each script activates its own env (main env for data/capture/masks, unsloth env for training),
# so start this driver with no venv active.
# qwen25-7b: data + capture + IWC masks already on disk from the earlier run -- retrain only if
# the new train_iwc_unsloth.sh hyperparameters (ga 32, lora r 16) should apply to this track too.
# bash scripts/data/data_qwen25-7b.sh
# bash scripts/capture/capture_qwen25-7b.sh
# bash scripts/masks/iwc_qwen25-7b.sh
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc-stable

bash scripts/data/data_qwen3-8b.sh
bash scripts/capture/capture_qwen3-8b.sh
bash scripts/masks/iwc_qwen3-8b.sh
bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc-stable

# ============================ EVAL =============================
# Leave the unsloth train env so each eval script activates the vLLM env (spectral-guided-learning).
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-unsloth-qwen25-7b iwc-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-unsloth-qwen25-7b iwc-stable-unsloth-qwen25-7b

bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-unsloth-qwen3-8b iwc-unsloth-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-unsloth-qwen3-8b iwc-stable-unsloth-qwen3-8b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json (regenerated from all results/<tag>/)
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" "${BASE}/src/compare_results.py"
