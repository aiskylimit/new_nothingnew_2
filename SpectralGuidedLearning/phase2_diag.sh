#!/usr/bin/env bash
# Why is a GPU idle during phase 2? Prints GPU use, the processes of each GPU, which phase-2 steps started /
# finished (and their exit codes), and the end of every phase-2 and driver log. Reads only; changes nothing.
#   bash phase2_diag.sh            # DIAG_LINES=40 for longer log tails
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
N="${DIAG_LINES:-25}"
hr() { printf '\n==================== %s ====================\n' "$*"; }

hr "GPUs"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
hr "GPU processes (pid -> GPU uuid)"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null
nvidia-smi --query-gpu=index,uuid --format=csv,noheader

hr "phase-2 / driver processes (pid, elapsed, CUDA_VISIBLE_DEVICES, command)"
for pid in $(pgrep -f 'project_commands_|sgl\.|robustness|torchrun|deepspeed|vllm' 2>/dev/null); do
  [[ -r "/proc/${pid}/environ" ]] || continue
  cvd="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"
  printf '%-8s %-12s cvd=%-6s %s\n' "${pid}" "$(ps -o etime= -p "${pid}" | tr -d ' ')" "${cvd:--}" \
    "$(ps -o args= -p "${pid}" | cut -c1-170)"
done

hr "phase2.out (start / finished lines: which steps ran, exit codes)"
if [[ -f phase2.out ]]; then grep -E 'start |finished|FAILED|phase 2 done' phase2.out | tail -40
  echo "--- last ${N} lines"; tail -n "${N}" phase2.out
else echo "phase2.out not found: phase 2 may not have been launched from $(pwd)"; fi

hr "phase-2 step logs (newest launch): last ${N} lines each"
latest="$(ls -t rebuttal/logs/phase2-*.log 2>/dev/null | head -1 | grep -oE '[0-9]{8}-[0-9]{6}')"
if [[ -z "${latest}" ]]; then echo "no rebuttal/logs/phase2-*.log"; else
  for f in $(ls -tr rebuttal/logs/phase2-*-"${latest}".log); do
    echo; echo "##### ${f}  ($(stat -c '%y' "${f}" | cut -d. -f1), $(wc -l < "${f}") lines)"
    grep -nE 'Traceback|Error|error:|FAILED|missing|failed|check fail|Killed|OOM|out of memory' "${f}" | tail -8
    echo "--- tail"; tail -n "${N}" "${f}"
  done
fi

hr "driver logs (results/{w4,hard,positional}/meta, newest of each)"
for d in results/w4/meta results/hard/meta results/positional/meta; do
  f="$(ls -t "${d}"/driver-*.log 2>/dev/null | head -1)"; [[ -n "${f}" ]] || { echo "${d}: no driver log"; continue; }
  echo; echo "##### ${f}"; tail -n "${N}" "${f}"
done

hr "run directories with an error in their logs"
for f in results/w4/runs/*/train/stderr.log results/w4/runs/*/eval.log results/hard/runs/*/train/stderr.log \
         results/hard/runs/*/eval.log; do
  [[ -f "${f}" ]] || continue
  e="$(grep -nE 'Traceback|Error|out of memory' "${f}" | tail -3)"; [[ -n "${e}" ]] && { echo "## ${f}"; echo "${e}"; }
done

hr "run status"
bash runs_status.sh
