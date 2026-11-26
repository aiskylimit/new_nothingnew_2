#!/usr/bin/env bash
# All runs still to do, one after the other (each script splits its own work over GPUs 0 and 1).
#   bash project_commands.sh                          # everything, in the order below
#   bash project_commands.sh positional vcore_asft    # only these, in this order
#   bash project_commands.sh --dry-run                # print every command, run nothing
#   GPUS="0 1" bash project_commands.sh               # GPUs (default 0 1)
# Finished stages are skipped, so after a crash just launch it again. One failed group does not stop the next.
#
#   rebuttal      answer gain + NLL + zero-shot on s1K-1.1, 3 models        project_commands_rebuttal.sh
#   answer_steps  loss on answer-containing steps only, s1K-1.1, 3 models    project_commands_answer_steps.sh
#   vcore_asft    ASFT and VCORE, 1.5B on the 966 traces                     project_commands_vcore_asft.sh
#   positional    positional-bias analyses A1-A4 + R6 (R7/R8 if the rule     project_commands_positional.sh
#                 asks), 1.5B on the 966 traces
#   w4            probe / segmentation robustness: scoring, rank stability,   project_commands_w4.sh
#                 12 runs (groups A, B, C), 1.5B on the 966 traces
# Outputs: rebuttal/ (first three), results/positional/ and results/w4/. Details: RUNS_REPORT.md.
set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

GROUPS_ALL=(rebuttal answer_steps vcore_asft positional w4)
selected=() flags=()
for arg in "$@"; do
  case "${arg}" in
    rebuttal|answer_steps|vcore_asft|positional|w4) selected+=("${arg}") ;;
    --dry-run) flags+=(--dry-run) ;;
    *) echo "unknown argument: ${arg} (groups: ${GROUPS_ALL[*]}, flag: --dry-run)" >&2; exit 2 ;;
  esac
done
[[ "${#selected[@]}" -gt 0 ]] || selected=("${GROUPS_ALL[@]}")

failed=()
for group in "${selected[@]}"; do
  echo "[$(date '+%F %T')] ===================== ${group} ====================="
  bash "project_commands_${group}.sh" ${flags[@]+"${flags[@]}"} || failed+=("${group}")
done

echo "[$(date '+%F %T')] finished: ${selected[*]}"
if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "FAILED: ${failed[*]}" >&2
  exit 1
fi
