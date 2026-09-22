#!/usr/bin/env bash
# SFT Long CoT driver -- DeepSeek-R1-Distill-Qwen-1.5B, FULL fine-tuning.
#   data (s1K-1.1 long CoT, all-ones mask) -> full-FT SFT (Unsloth, 1 GPU) -> eval -> compare.
# Eval follows P-ALIGN/src/test.py: thinking OFF, n=3, T=0.6, top_p=0.9, repetition_penalty=1.05,
# max_tokens=4096 over AIME24/AIME25/AMC12/MATH500 (Pass@1 + Pass@3).
# The spectral/IWC arms of this track live in scripts/{capture,masks,spectral}/ and are NOT run
# here; this driver is the plain SFT baseline end to end.
#   GPUS=0 bash project_commands_r1-qwen-1.5b.sh
# Comment out any line you don't want to run.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

# Which GPU(s) each phase runs on (space-separated ids). Training is single-GPU (Unsloth OSS
# takes the first id); eval uses the whole list.
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"

# Thinking mode for the SUPERVISION format. true (default) = the long CoT is supervised inside
# R1-Distill's native <think> block, which is what its template opens and what s1K-1.1's
# trajectories already look like; false closes the block and supervises the CoT as plain prose.
export ENABLE_THINKING="${ENABLE_THINKING:-true}"

# ============================ TRAIN ============================
# Each script activates its own venv (data: spectral_guided_learning, train: ..._train), so
# leave the shell env clean here.
bash scripts/data/data_r1-qwen-1.5b.sh
bash scripts/sft/sft_r1-qwen-1.5b.sh

# ============================ EVAL =============================
# Leave any train env so the eval script activates the vLLM env. Thinking is OFF at eval
# regardless of how the data was built -- override with ENABLE_THINKING_EVAL=true.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
ENABLE_THINKING="${ENABLE_THINKING_EVAL:-false}" \
  bash scripts/eval/eval_r1-qwen-1.5b.sh checkpoints/vanilla-r1-qwen-1.5b vanilla-r1-qwen-1.5b

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
"${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python" "${BASE}/src/compare_results.py"
