#!/usr/bin/env bash
# Phase 3: run 1 (ALERD = answer-gain IWC-Stable, lambda 0.5, s1K-1.1) for the two larger students, LoRA r16:
#   GPU 0: run 4   qwen25-7b  (configs/sgl/qwen25-7b-s1k/iwc-gain-l05-lora.yaml)
#   GPU 1: run 7   qwen3-8b   (configs/sgl/qwen3-8b-s1k/iwc-gain-l05-lora.yaml)
# ALG arm only (no SFT / zero-shot). The two lanes write disjoint data dirs (data/<model>-s1k/), so they run
# side by side. Answer gain is scored by the student itself, so it is several times slower than run 1's.
# Each lane starts only if: its GPU uses at most MAX_USED_MIB (10000), the run is neither done nor
# running, and for GPU 0, run 1 is finished (its eval summary exists; the GPU is briefly empty between stages).
# Finished stages are skipped: after a crash, launch it again. Foreground, waits for both lanes.
#   bash phase3_alerd_7b_8b.sh
#   PHASE3_GPUS="2 3" bash phase3_alerd_7b_8b.sh   # other GPU ids, in the order of the lanes above
# Logs: rebuttal/logs/phase3-gpu<g>-<model>-<time>.log (+ each driver's driver-<time>.log).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

read -ra G <<< "${PHASE3_GPUS:-0 1}"
[[ "${#G[@]}" == 2 ]] || { echo "PHASE3_GPUS needs 2 GPU ids, got: ${G[*]}" >&2; exit 2; }
RUN1_EVAL=rebuttal/results/iwc-gain-l05-r1-qwen-1.5b-s1k/summary.json
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p rebuttal/logs

# The server keeps small processes on every GPU even when it is free (about 4.7 GB in nvidia-smi), so "free"
# is judged by used memory, not by the absence of processes.
MAX_USED_MIB="${MAX_USED_MIB:-10000}"

# ready <gpu> <model> <run no>: prints why not and returns 1 when the lane must not start
ready() {
  local gpu="$1" model="$2" run="$3" ok=0 used running
  local cfg="${model}-s1k/iwc-gain-l05-lora" eval="rebuttal/results/iwc-gain-l05-lora-${model}-s1k/summary.json"
  if ! used=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits 2>&1) ||
     ! [[ "${used}" =~ ^[0-9]+$ ]]; then
    echo "STOP run ${run}: nvidia-smi failed for GPU ${gpu}: ${used:0:100}"; return 1
  fi
  (( used <= MAX_USED_MIB )) || { echo "STOP run ${run}: GPU ${gpu} is in use (${used} MiB)"; ok=1; }
  [[ -f "${eval}" ]] && { echo "STOP run ${run}: already done (${eval})"; ok=1; }
  running=$(pgrep -af "sgl\.cli run ${cfg//./\\.}( |$)")
  [[ -z "${running}" ]] || { echo "STOP run ${run}: already running:"; echo "${running}"; ok=1; }
  return "${ok}"
}

LANES=("${G[0]}|qwen25-7b|4" "${G[1]}|qwen3-8b|7")
go=()
for lane in "${LANES[@]}"; do
  IFS='|' read -r gpu model run <<< "${lane}"
  extra=0
  if [[ "${gpu}" == 0 && ! -f "${RUN1_EVAL}" ]]; then
    echo "STOP run ${run}: run 1 is not finished on GPU 0 (${RUN1_EVAL} missing)"; extra=1
  fi
  ready "${gpu}" "${model}" "${run}" && [[ "${extra}" == 0 ]] && go+=("${lane}")
done
[[ "${#go[@]}" -gt 0 ]] || { echo "nothing started"; exit 1; }

trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops both lanes
pids=() names=()
for lane in "${go[@]}"; do
  IFS='|' read -r gpu model run <<< "${lane}"
  log="rebuttal/logs/phase3-gpu${gpu}-${model}-${STAMP}.log"
  echo "[$(date '+%F %T')] start run ${run} (${model} ALERD LoRA, s1K-1.1) on GPU ${gpu} -> ${log}"
  GPUS="${gpu}" MODELS="${model}" ARMS=alg bash project_commands_rebuttal.sh > "${log}" 2>&1 &
  pids+=($!); names+=("run ${run} (${model}, GPU ${gpu})")
  sleep 2   # each driver names its own log driver-<second>.log: keep them apart
done
failed=()
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" && echo "[$(date '+%F %T')] ${names[$i]} finished" || failed+=("${names[$i]}")
done
[[ "${#failed[@]}" == 0 ]] || { printf 'FAILED: %s (see rebuttal/logs/)\n' "${failed[@]}" >&2; exit 1; }
echo "[$(date '+%F %T')] phase 3 done: rebuttal/results/comparison-table.md"
