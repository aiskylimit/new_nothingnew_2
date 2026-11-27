#!/usr/bin/env bash
# Phase 2 of RUNS_REPORT.md (2026-11-27): what is left after GPUs 0, 1, 6 were freed. GPU 7 keeps running
# run 14 (VCORE) from phase 1 and is not touched. Runs 2-9, 11, 12, 16, 17, 19, 21-29, 31-33 are no longer planned.
#   prelude (GPU 0, minutes): positional analyses A1-A4, ALG-hard step selection (both CPU after `prepare`)
#   GPU 0     run 1  r1-qwen-1.5b on s1K-1.1, ALERD                  (project_commands_rebuttal.sh, ARMS=alg)
#             run 10 r1-qwen-1.5b on s1K-1.1, answer-steps-only      (same data dir as run 1: after it)
#   GPU 1, 6  W4 with one other probe and one other segmentation (VARIANTS="P1/S0 P0/S1"): scoring +
#             rank stability, runs 18 (P1/S0) and 20 (P0/S1), one per GPU (group C, runs 26-29, dropped); then
#   GPU 1     run 30 ALG-hard H59 only (runs 31-33, R59 / H30 / ALG-H59, dropped)
#   GPU 6     run 13 ASFT, 966 traces, then run 15 positional R6 and the decision rule (R7/R8, runs 16-17,
#             dropped: FOLLOWUPS=0)
#   end (all three GPUs): ALG-hard post-hoc NLL (seed 42 only: no group C) and report, comparison tables
# The prelude runs before the lanes so only one process ever builds data/r1-qwen-1.5b-palign/ (the lanes
# then only read it); s1K runs use data/r1-qwen-1.5b-s1k/, which only the first GPU touches.
#   bash project_commands_phase2.sh             # everything above, waits for both lanes
#   bash project_commands_phase2.sh --dry-run   # print every command, run nothing
#   PHASE2_GPUS="0 1 6" bash project_commands_phase2.sh   # other GPU ids, in the roles of 0, 1, 6 above
# Finished steps are skipped: after a crash launch it again. Logs: rebuttal/logs/phase2-<step>-<time>.log
# (plus each driver's own logs under rebuttal/logs/, results/{positional,w4,hard}/meta/).
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

read -ra G <<< "${PHASE2_GPUS:-0 1 6}"
[[ "${#G[@]}" == 3 ]] || { echo "PHASE2_GPUS needs 3 GPU ids, got: ${G[*]}" >&2; exit 2; }
GA="${G[0]}" G1="${G[1]}" G2="${G[2]}" GB="${G[1]} ${G[2]}" GALL="${G[*]}"
FLAGS=("$@")
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p rebuttal/logs

step() {  # <name> <command...>
  local name="$1"; shift
  echo "[$(date '+%F %T')] start ${name}"
  "$@" > "rebuttal/logs/phase2-${name}-${STAMP}.log" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] ${name} finished, exit ${status}"
  return "${status}"
}

failed=()
# ---- prelude: analyses that need no training; also builds the shared 966-trace data once
GPUS="${GA}" step positional-analyze bash project_commands_positional.sh analyze ${FLAGS[@]+"${FLAGS[@]}"} \
  || failed+=("positional analyses")
GPUS="${GA}" step hard-prepare bash project_commands_hard.sh prepare ${FLAGS[@]+"${FLAGS[@]}"} \
  || failed+=("hard selection")

# ---- two lanes side by side
lane_a() {
  local s=0
  GPUS="${GA}" MODELS=r1-qwen-1.5b ARMS=alg step gpu${GA}-run1-alerd-s1k \
    bash project_commands_rebuttal.sh "$@" || s=1
  GPUS="${GA}" MODELS=r1-qwen-1.5b step gpu${GA}-run10-answer-steps \
    bash project_commands_answer_steps.sh "$@" || s=1
  return "${s}"
}
lane_b() {
  local s=0 p1 p2
  # group A alone: with VARIANTS it is just P1/S0 and P0/S1; no group C (seeds 43/44)
  GPUS="${GB}" VARIANTS="P1/S0 P0/S1" step "gpus${G1}${G2}-runs18-20-w4" \
    bash project_commands_w4.sh all A "$@" || s=1
  # then the two GPUs split: H59 on one, ASFT and R6 on the other
  GPUS="${G1}" HARD_ARMS=h59 step "gpu${G1}-run30-hard-h59" bash project_commands_hard.sh train A "$@" & p1=$!
  ( t=0
    GPUS="${G2}" ARMS=sft-asft step "gpu${G2}-run13-asft" bash project_commands_vcore_asft.sh "$@" || t=1
    GPUS="${G2}" FOLLOWUPS=0 step "gpu${G2}-run15-positional" bash project_commands_positional.sh all "$@" || t=1
    exit "${t}" ) & p2=$!
  wait "${p1}" || s=1
  wait "${p2}" || s=1
  return "${s}"
}
trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops both lanes
lane_a ${FLAGS[@]+"${FLAGS[@]}"} & pa=$!
lane_b ${FLAGS[@]+"${FLAGS[@]}"} & pb=$!
wait "${pa}" || failed+=("gpu${GA}: runs 1, 10")
wait "${pb}" || failed+=("gpus ${GB}: runs 13, 15, 18, 20, 30")

# ---- after both lanes: NLL of the uniform / ALG seed-42 checkpoints (E0 / E7), then the reports
GPUS="${GALL}" NLL_SEEDS=42 step hard-nll-posthoc bash project_commands_hard.sh nll_posthoc ${FLAGS[@]+"${FLAGS[@]}"} \
  || failed+=("hard nll_posthoc")
step hard-report bash project_commands_hard.sh report ${FLAGS[@]+"${FLAGS[@]}"} || failed+=("hard report")
if [[ " ${FLAGS[*]-} " != *" --dry-run "* ]]; then
  PY="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python"
  PYTHONPATH="${BASE}/src" "${PY}" -m sgl.eval.compare --results-dir rebuttal/results
  PYTHONPATH="${BASE}/src" "${PY}" -m sgl.diagnostics.efficiency
fi
if [[ "${#failed[@]}" -gt 0 ]]; then
  printf 'FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[$(date '+%F %T')] phase 2 done: rebuttal/results/, results/positional/, results/w4/, results/hard/"
