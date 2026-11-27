#!/usr/bin/env bash
# Run 10 (1.5B answer-steps-only, s1K-1.1) on GPU 1 while GPU 0 is still busy with run 1.
# Run 10 needs no answer-gain scoring (stages prepare, answer_mask, train, eval) and prepare is already
# done, so it can run beside run 1. Every stage has a `creates` marker: when lane A of phase 2 reaches
# run 10 later, it skips everything -- as long as lane A is only released after run 10 has FINISHED
# (otherwise it would retrain into the same checkpoint dir).
# Starts only if: run 30 is finished, none of this project's processes is on GPU 1, run 10 is neither
# done nor running. Runs in the foreground.
#   bash phase2_run10_gpu1.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GPU=1
RUN30=results/hard/runs/r1q1.5b__h59__P0__S0__seed42/run_check.json
RUN10_TRAIN=checkpoints/ansstep-mention-r1-qwen-1.5b-s1k/run-summary.json
RUN10_EVAL=rebuttal/results/ansstep-mention-r1-qwen-1.5b-s1k/summary.json

ok=1
[[ -f "${RUN30}" ]] || { echo "STOP: run 30 is not finished (${RUN30} missing)"; ok=0; }
ours=""
if ! apps=$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>&1); then
  echo "STOP: nvidia-smi failed, cannot check GPU ${GPU}: ${apps:0:100}"; exit 1
fi
for pid in ${apps}; do
  [[ "${pid}" =~ ^[0-9]+$ ]] || { echo "STOP: unexpected nvidia-smi output: ${apps:0:100}"; exit 1; }
  args=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)
  [[ "${args}" == *spectral_guided_learning* || "${args}" == *sgl.* ]] && ours+="  ${pid} ${args:0:120}"$'\n'
done
[[ -z "${ours}" ]] || { echo "STOP: this project still has processes on GPU ${GPU}:"; printf '%s' "${ours}"; ok=0; }
[[ -f "${RUN10_EVAL}" ]] && { echo "STOP: run 10 is already done (${RUN10_EVAL})"; ok=0; }
running=$(pgrep -af '^bash project_commands_answer_steps\.sh|sgl\.cli run r1-qwen-1\.5b-s1k/answer-steps-only')
[[ -z "${running}" ]] || { echo "STOP: run 10 is already running:"; echo "${running}"; ok=0; }
[[ "${ok}" == 1 ]] || exit 1

[[ -f "${RUN10_TRAIN}" ]] && echo "run 10 already trained: only eval will run"
echo "[$(date '+%F %T')] start run 10 on GPU ${GPU}"
GPUS="${GPU}" MODELS=r1-qwen-1.5b bash project_commands_answer_steps.sh
