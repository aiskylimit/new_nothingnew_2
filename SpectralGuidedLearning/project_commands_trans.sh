#!/usr/bin/env bash
# L_trans experiment driver (method B: next-step representation prediction), Qwen3-8B track.
# Config 4 of the experiment table: SFT + L_trans, lambda = 0.3. Same seed/LR/scheduler/LoRA as the
# SFT baseline (scripts/sft/sft_qwen3-8b.sh); the only change is the added objective.
# Paths are disjoint from the other drivers (data/qwen3-8b/train-*-trans.jsonl, checkpoints/trans-*,
# logs/trans-*, results/trans-*); only compare_results.py rewrites the shared summary files.
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) every phase script runs on (space-separated ids). Override: GPUS="0" ./project_commands_trans.sh
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0 1}"

# ============================ DATA ============================
# Needs data/qwen3-8b/train-segmented.jsonl + train-vanilla.jsonl already on disk (data_qwen3-8b.sh,
# masks_qwen3-8b.sh). Adds \n\n-step fields; the NLL mask is copied untouched. Seconds, CPU only.
bash scripts/trans/build_trans_qwen3-8b.sh vanilla
# bash scripts/trans/build_trans_qwen3-8b.sh spectral        # for config 5 (SGL + L_trans)

# ============================ TRAIN ============================
# Effective batch 32 regardless of GPU count (ga = 32 / n_gpu). Watch in logs/trans-*.log:
#   loss_trans / trans_cos      should fall / rise; trans_cos must clear trans_copy_cos
#   trans_raw_step_cos -> 1     step representations collapsing
#   trans_ztilde_norm -> 0      step-specific part vanishing
#   grad_trans_ratio            lambda*||grad L_trans|| / ||grad L_NLL|| on LoRA; persistently > 0.3-0.5 => lower lambda
bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.3                 # config 4
# bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.1               # config 3
# bash scripts/trans/train_trans_qwen3-8b.sh spectral 0.3              # config 5 (best lambda)
# bash scripts/trans/train_trans_qwen3-8b.sh vanilla 0.3 shuffle       # config 6 (control)

# ============================ EVAL =============================
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
# Adapter-only checkpoint like every other arm; trans_predictor.pt next to it is ignored by vLLM.
bash scripts/eval/eval_qwen3-8b.sh checkpoints/trans-vanilla-l0.3-qwen3-8b trans-vanilla-l0.3-qwen3-8b

# =========================== COMPARE ==========================
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" "${BASE}/src/compare_results.py"
