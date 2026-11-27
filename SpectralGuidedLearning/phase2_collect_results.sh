#!/usr/bin/env bash
# Gather the phase-2 results (no model weights) for fetch jobs, which take one folder of at most 25 MiB:
#   _fetch/sgl-phase2/          every file under 512 KiB, uncompressed (summaries, tables, metrics, short logs)
#   _fetch/sgl-phase2-partNN/   ALL files as one tar.gz, cut into 24 MiB pieces, one piece per folder
# One fetch job per folder:  #2 -f-<repo>/_fetch/sgl-phase2/ +a  (then -part01/, -part02/, ...)
# Reassemble locally:        cat sgl-phase2-part*/full.tar.gz.part* | tar -xzf -
# Reads the run dirs only; _fetch/sgl-phase2* is rebuilt from scratch on every call (rerun after run 1).
#   bash phase2_collect_results.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
FETCH="${FETCH:-$(pwd)/_fetch}"            # inside this repo (git-ignored)
LIMIT=26214400                             # fetch job limit (25 MiB)
PIECE=24M                                  # archive piece size, under the limit
PATHS=(rebuttal/results rebuttal/logs results/w4 results/hard results/positional
       experiments/answer_gain/scores/answer_gain.json runs_status.sh)

present=()
for p in "${PATHS[@]}"; do
  if [[ -e "${p}" ]]; then present+=("${p}"); else echo "WARNING: ${p} not found, skipped"; fi
done
[[ "${#present[@]}" -gt 0 ]] || { echo "nothing to collect" >&2; exit 1; }
[[ "${FETCH}" == */_fetch ]] || { echo "refusing to clear ${FETCH}: not a _fetch folder" >&2; exit 1; }
rm -rf "${FETCH}"/sgl-phase2 "${FETCH}"/sgl-phase2-part* && mkdir -p "${FETCH}/sgl-phase2"
CORE="${FETCH}/sgl-phase2"

# every result file except weights and optimizer state (epoch checkpoints are several GB per run)
files() {  # [extra find tests...]
  find "${present[@]}" \( -name checkpoints -o -name checkpoint -o -name 'checkpoint-*' \) -prune -o \
    -type f ! -name '*.safetensors' ! -name '*.bin' ! -name '*.pt' ! -name '*.pth' "$@" -print0
}

# 1) small files, readable as they are
files -size -512k | tar --null -T - -cf - | tar -xf - -C "${CORE}" || { echo "copy failed" >&2; exit 1; }
bash runs_status.sh > "${CORE}/runs_status.txt" 2>&1
date '+%F %T' > "${CORE}/collected_at.txt"
files ! -size -512k | xargs -0 -r du -k | sort -rn | awk -F'\t' '{printf "%8.1f MiB  %s\n", $1/1024, $2}' \
  > "${CORE}/big_files_in_parts.txt"

# 2) everything, compressed and cut into pieces
tmp="$(mktemp -d "${FETCH}/.tmp.XXXX")"
files | tar --null -T - -czf - | split -b "${PIECE}" -d -a 2 - "${tmp}/full.tar.gz.part" \
  || { rm -rf "${tmp}"; echo "archive failed" >&2; exit 1; }
n=0
for piece in "${tmp}"/full.tar.gz.part*; do
  n=$((n + 1)); dir="${FETCH}/sgl-phase2-part$(printf '%02d' "${n}")"
  mkdir -p "${dir}" && mv "${piece}" "${dir}/"
done
rm -rf "${tmp}"
echo "${n} archive pieces; reassemble: cat sgl-phase2-part*/full.tar.gz.part* | tar -xzf -" > "${CORE}/README.txt"

echo "== fetch folders (limit ${LIMIT} B each):"
status=0
for dir in "${CORE}" "${FETCH}"/sgl-phase2-part*; do
  bytes=$(du -sb "${dir}" | cut -f1)
  flag=ok; [[ "${bytes}" -le "${LIMIT}" ]] || { flag="TOO BIG"; status=1; }
  printf '  %-8s %6.1f MiB  #2 -f-%s/ +a\n' "${flag}" "$(awk -v b="${bytes}" 'BEGIN{print b/1048576}')" "${dir}"
done
echo "== largest files (only in the parts):"; head -8 "${CORE}/big_files_in_parts.txt"
exit "${status}"
