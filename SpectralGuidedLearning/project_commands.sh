#!/usr/bin/env bash
# Spectral experiment driver -- TRAIN everything, then EVAL, then compare.
# Models: qwen25-7b, qwen3-8b (P-ALIGN's two student models). Dataset: s1K-1.1.
# Comment out any line you don't want to run.
#
# The r1-qwen-1.5b / r1-qwen-7b scripts are still in scripts/ but are out of the driver:
# they train on LIMO and are not part of the P-ALIGN comparison.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) every phase script runs on (space-separated ids). Override: GPUS="0 1" ./project_commands.sh
export GPUS="${GPUS:-1}"

# ============================ TRAIN ============================
# per model: data -> capture (spectral + entropy) -> masks -> four matched SFT arms.
# IWC and IWC-Stable share exactly the spectral-selected token set; only weights differ.
source /mnt/local/uvenvs/spectral_guided_learning/bin/activate
bash scripts/data/data_qwen25-7b.sh
bash scripts/capture/capture_qwen25-7b.sh
bash scripts/masks/masks_qwen25-7b.sh
bash scripts/masks/iwc_qwen25-7b.sh
# bash scripts/spectral/spectral_qwen25-7b.sh
# bash scripts/sft/sft_qwen25-7b.sh
bash scripts/iwc/train_iwc.sh qwen25-7b iwc
bash scripts/iwc/train_iwc.sh qwen25-7b iwc-stable

# 8B tạm tắt để thử qwen25-7b trước.
# bash scripts/data/data_qwen3-8b.sh
# bash scripts/capture/capture_qwen3-8b.sh
# bash scripts/masks/masks_qwen3-8b.sh
# bash scripts/masks/iwc_qwen3-8b.sh
# bash scripts/spectral/spectral_qwen3-8b.sh
# bash scripts/sft/sft_qwen3-8b.sh
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc-stable

# ============================ EVAL =============================
# only the iwc / iwc-stable checkpoints (vanilla/spectral training is commented out above)

# bash scripts/eval/eval_qwen25-7b.sh
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/vanilla-qwen25-7b vanilla-qwen25-7b
bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-qwen25-7b iwc-qwen25-7b
bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-qwen25-7b iwc-stable-qwen25-7b

# bash scripts/eval/eval_qwen3-8b.sh
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-qwen3-8b vanilla-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-qwen3-8b iwc-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-qwen3-8b iwc-stable-qwen3-8b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
python "${BASE}/src/compare_results.py"
