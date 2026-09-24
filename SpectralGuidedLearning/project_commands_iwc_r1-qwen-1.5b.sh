#!/usr/bin/env bash
# IWC driver -- DeepSeek-R1-Distill-Qwen-1.5B.
#   data -> capture (spectral + entropy) -> IWC weights -> IWC SFT (full + LoRA) -> eval -> compare.
# IWC-Stable keeps train-spectral's exact token selection and only reweights it, so each arm is
# matched to its spectral counterpart from project_commands_spectral_r1-qwen-1.5b.sh:
#   iwc-stable-full-r1-qwen-1.5b <-> spectral-full-r1-qwen-1.5b
#   iwc-stable-lora-r1-qwen-1.5b <-> spectral-lora-r1-qwen-1.5b
#   GPUS=0 bash project_commands_iwc_r1-qwen-1.5b.sh
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

# ============================ TRAIN ============================
# Shared with the spectral driver; skipped when already on disk. The parquet must carry
# step_entropies -- build_iwc_datasets.py stops with a clear error if it predates that.
[[ -f data/r1-qwen-1.5b/train-segmented.jsonl ]]     || bash scripts/data/data_r1-qwen-1.5b.sh
[[ -f data/r1-qwen-1.5b/spectral-strengths.parquet ]] || bash scripts/capture/capture_r1-qwen-1.5b.sh

# IWC-Stable weights (Eq. 13-16), same gentle setting as the qwen3-8b track (project_commands2.sh):
# lambda 0.5, tau 2.0, clip 1.0. Token mass is preserved per sample -- the build log must print
# mean weighted/selected mass=1.000000 for iwc-stable.
export IWC_INTERPOLATION="${IWC_INTERPOLATION:-0.5}"
export IWC_TEMPERATURE="${IWC_TEMPERATURE:-2.0}"
export IWC_CLIP="${IWC_CLIP:-1.0}"
bash scripts/masks/iwc_r1-qwen-1.5b.sh

bash scripts/iwc/train_iwc_r1-qwen-1.5b.sh full iwc-stable
bash scripts/iwc/train_iwc_r1-qwen-1.5b.sh lora iwc-stable
# bash scripts/iwc/train_iwc_r1-qwen-1.5b.sh full iwc   # Eq. 11 baseline: not mass-preserving

# ============================ EVAL =============================
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/iwc-stable-full-r1-qwen-1.5b iwc-stable-full-r1-qwen-1.5b
bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/iwc-stable-lora-r1-qwen-1.5b iwc-stable-lora-r1-qwen-1.5b
# bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/iwc-full-r1-qwen-1.5b iwc-full-r1-qwen-1.5b

# =========================== COMPARE ==========================
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" "${BASE}/src/compare_results.py"
