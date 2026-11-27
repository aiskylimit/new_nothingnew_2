#!/usr/bin/env bash
# Phase 1 of RUNS_REPORT.md: runs 1-9 (s1K-1.1, 3 students) and 13-14 (ASFT, VCORE), one run per GPU,
# each GPU pinned to its runs:
#   GPU 0: runs 1-3   r1-qwen-1.5b on s1K-1.1 (ALG, SFT, zero-shot), then run 13 (ASFT, 1.5B, 966 traces)
#   GPU 1: runs 4-6   qwen25-7b on s1K-1.1 (ALG, SFT, zero-shot), LoRA r16
#   GPU 6: runs 7-9   qwen3-8b on s1K-1.1 (ALG, SFT, zero-shot), LoRA r16
#   GPU 7: run 14     VCORE (1.5B, 966 traces): about 3-4x an SFT run, so it has a GPU to itself
# ASFT waits for the 1.5B s1K runs because that student is the lightest of the three s1K tracks.
# The four lanes write disjoint data dirs (data/<model>-s1k/, data/r1-qwen-1.5b-palign/).
#   bash project_commands_phase1.sh             # start all four lanes, wait for them
#   bash project_commands_phase1.sh --dry-run   # print every command, run nothing
#   PHASE1_GPUS="0 1 6 7" bash project_commands_phase1.sh   # other GPU ids, same order as the lanes above
# Finished stages are skipped: after a crash launch it again. Logs: rebuttal/logs/phase1-<lane>-<time>.log
# (plus each driver's own driver-<time>.log); scores: rebuttal/results/comparison-table.md.
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

read -ra G <<< "${PHASE1_GPUS:-0 1 6 7}"
[[ "${#G[@]}" == 4 ]] || { echo "PHASE1_GPUS needs 4 GPU ids, got: ${G[*]}" >&2; exit 2; }
FLAGS=("$@")
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p rebuttal/logs

lane() {  # <name> <command...>
  local name="$1"; shift
  echo "[$(date '+%F %T')] start ${name}"
  "$@" > "rebuttal/logs/phase1-${name}-${STAMP}.log" 2>&1
  local status=$?
  echo "[$(date '+%F %T')] ${name} finished, exit ${status}"
  return "${status}"
}

pids=() names=()
lane "gpu${G[0]}-r1-qwen-1.5b-s1k+asft" bash -c '
  GPUS="$1" MODELS=r1-qwen-1.5b bash project_commands_rebuttal.sh "${@:2}"; s1=$?
  GPUS="$1" ARMS=sft-asft bash project_commands_vcore_asft.sh "${@:2}"; s2=$?
  exit $(( s1 || s2 ))' _ "${G[0]}" ${FLAGS[@]+"${FLAGS[@]}"} &
pids+=($!); names+=("gpu${G[0]}: runs 1-3 + 13")
GPUS="${G[1]}" MODELS=qwen25-7b lane "gpu${G[1]}-qwen25-7b-s1k" bash project_commands_rebuttal.sh ${FLAGS[@]+"${FLAGS[@]}"} &
pids+=($!); names+=("gpu${G[1]}: runs 4-6")
GPUS="${G[2]}" MODELS=qwen3-8b lane "gpu${G[2]}-qwen3-8b-s1k" bash project_commands_rebuttal.sh ${FLAGS[@]+"${FLAGS[@]}"} &
pids+=($!); names+=("gpu${G[2]}: runs 7-9")
GPUS="${G[3]}" ARMS=sft-vcore lane "gpu${G[3]}-vcore" bash project_commands_vcore_asft.sh ${FLAGS[@]+"${FLAGS[@]}"} &
pids+=($!); names+=("gpu${G[3]}: run 14")

trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops every lane
failed=()
for i in "${!pids[@]}"; do wait "${pids[$i]}" || failed+=("${names[$i]}"); done

# each driver rewrote the table when it ended; rebuild it once more now that every lane is done
if [[ " ${FLAGS[*]-} " != *" --dry-run "* ]]; then
  PY="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python"
  PYTHONPATH="${BASE}/src" "${PY}" -m sgl.eval.compare --results-dir rebuttal/results
  PYTHONPATH="${BASE}/src" "${PY}" -m sgl.diagnostics.efficiency
fi
if [[ "${#failed[@]}" -gt 0 ]]; then
  printf 'FAILED: %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "[$(date '+%F %T')] phase 1 done: rebuttal/results/comparison-table.md"
