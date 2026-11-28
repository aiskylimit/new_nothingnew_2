#!/usr/bin/env bash
# Progress of phase 3 (runs 4 and 7: ALERD LoRA on s1K-1.1, 7B on GPU 0, 8B on GPU 1). Reads files and
# ps only, changes nothing; safe while the runs are going.
#   bash phase3_status.sh            # LINES=15 bash phase3_status.sh for longer log tails
# Per run: each stage of the config with its `creates` marker (done + time, or -), the stage running now,
# the last lines of its stage log (progress bars' \r turned into lines), and the lane log's last lines.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}/bin/python"
[[ -x "${PY}" ]] || PY=python3
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
LINES="${LINES:-8}"
STAGES=(prepare capture answer_gain gain_signal weights answer_stats train eval)

echo "== $(date '+%F %T')"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 0,1 2>&1 | sed 's/^/  GPU /'

for entry in "4|qwen25-7b" "7|qwen3-8b"; do
  IFS='|' read -r run model <<< "${entry}"
  cfg="${model}-s1k/iwc-gain-l05-lora" name="iwc-gain-l05-lora-${model}-s1k"
  echo
  echo "======== run ${run}: ${model} (${cfg})"
  # "<stage> <creates>" for each stage of the resolved config
  markers=$("${PY}" -m sgl.cli show "${cfg}" 2>/dev/null | awk '
    /^stages:/ { in_st = 1; next }
    in_st && /^[^ ]/ { in_st = 0 }
    in_st && /^  [a-z_]+:$/ { stage = $1; sub(":", "", stage) }
    in_st && /^    creates:/ { print stage, $2 }')
  [[ -n "${markers}" ]] || echo "  (could not read the config with ${PY})"
  current=""
  for stage in "${STAGES[@]}"; do
    file=$(awk -v s="${stage}" '$1 == s { print $2 }' <<< "${markers}")
    if [[ -z "${file}" ]]; then state="(no marker)"
    elif [[ -f "${file}" ]]; then state="done $(date -r "${file}" '+%m-%d %H:%M')"
    else state="-"; [[ -z "${current}" ]] && current="${stage}"
    fi
    printf '  %-13s %-18s %s\n' "${stage}" "${state}" "${file}"
  done
  echo "  next / running stage: ${current:-all done}"

  procs=$(pgrep -af "sgl\.cli run ${cfg//./\\.}( |$)" | cut -c1-150)
  echo "  sgl.cli process: ${procs:-(none)}"
  if [[ -n "${current}" ]]; then
    log="rebuttal/logs/${name}-${current}.log"
    if [[ -f "${log}" ]]; then
      echo "  -- ${log} (modified $(date -r "${log}" '+%m-%d %H:%M')):"
      tail -c 200000 "${log}" | tr '\r' '\n' | grep -v '^\s*$' | tail -n "${LINES}" | cut -c1-200 | sed 's/^/     /'
    else
      echo "  -- no stage log yet (${log})"
    fi
  fi
  lane=$(ls -t rebuttal/logs/phase3-gpu*-"${model}"-*.log 2>/dev/null | head -1)
  if [[ -n "${lane}" ]]; then
    echo "  -- lane log ${lane}:"
    tail -c 100000 "${lane}" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 4 | cut -c1-200 | sed 's/^/     /'
  else
    echo "  -- no lane log yet (phase3_alerd_7b_8b.sh not started?)"
  fi
done
