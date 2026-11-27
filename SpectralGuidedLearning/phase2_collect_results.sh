#!/usr/bin/env bash
# Copy the phase-2 results (no model weights) into one folder, so a single fetch job can take them:
#   #2 -f-/mnt/local/aiskylimit_new_nothingnew_2/SpectralGuidedLearning/_fetch/sgl-phase2/ +a
# Reads the run dirs only; the copy is rebuilt from scratch on every call (rerun it after run 1 finishes).
#   bash phase2_collect_results.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="${OUT:-$(pwd)/_fetch/sgl-phase2}"   # inside this repo (git-ignored)
PATHS=(rebuttal/results rebuttal/logs results/w4 results/hard results/positional
       experiments/answer_gain/scores/answer_gain.json runs_status.sh)

present=()
for p in "${PATHS[@]}"; do
  if [[ -e "${p}" ]]; then present+=("${p}"); else echo "WARNING: ${p} not found, skipped"; fi
done
[[ "${#present[@]}" -gt 0 ]] || { echo "nothing to collect" >&2; exit 1; }

[[ "${OUT}" == */_fetch/* ]] || { echo "refusing to clear ${OUT}: not under a _fetch/ folder" >&2; exit 1; }
rm -rf "${OUT}" && mkdir -p "${OUT}"
# weights and optimizer state stay on the server: epoch checkpoints are several GB per run
tar -cf - --exclude='checkpoints' --exclude='checkpoint' --exclude='checkpoint-*' \
    --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' --exclude='*.pth' "${present[@]}" \
  | tar -xf - -C "${OUT}" || { echo "copy failed" >&2; exit 1; }
bash runs_status.sh > "${OUT}/runs_status.txt" 2>&1
date '+%F %T' > "${OUT}/collected_at.txt"

echo "collected -> ${OUT}"
du -sh "${OUT}"
du -sh "${OUT}"/*/* 2>/dev/null | sort -h | tail -8
echo "fetch job header:  #2 -f-${OUT}/ +a"
