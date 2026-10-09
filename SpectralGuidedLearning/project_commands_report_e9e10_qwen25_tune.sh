#!/usr/bin/env bash
# One-command report for E9/E10 and the Qwen2.5-7B tau=1, batch-8 tuning grid.
# Safe to rerun while jobs are still running: unfinished arms are displayed as "missing".
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
if [[ ! -f "${PROJECT_ENV}/bin/activate" ]]; then
  echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV=/path/to/venv)" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${PROJECT_ENV}/bin/activate"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

LOG_DIR="logs/reports"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/report-$(date +%Y%m%d-%H%M%S).log"
ARMS=(
  iwc-gain-l25-t1-c2-b8-lora iwc-gain-l50-t1-c2-b8-lora
  iwc-gain-l75-t1-c2-b8-lora iwc-gain-l100-t1-c2-b8-lora
  iwc-gain-l25-t1-c1-b8-lora iwc-gain-l50-t1-c1-b8-lora
  iwc-gain-l75-t1-c1-b8-lora iwc-gain-l25-t1-c3-b8-lora
  iwc-gain-l50-t1-c3-b8-lora iwc-gain-l75-t1-c3-b8-lora
)

{
  echo "=== Report generated $(date -Is) ==="
  echo
  echo "=== E9/E10 normalization ablation ==="
  python -m sgl.eval.answer_gain_summary --results-dir results --out-dir results \
    --track r1-qwen-1.5b-palign
  echo
  echo "=== Qwen2.5-7B tuning: tau=1, effective batch=8 ==="
  python -m sgl.eval.qwen25_tuning_summary --results-dir results \
    --out-dir results/qwen25-tuning-tau1-b8 --arms "${ARMS[@]}"
  echo
  echo "Saved reports:"
  echo "  results/ablation_summary.md"
  echo "  results/normalization_diagnostics.csv"
  echo "  results/qwen25-tuning-tau1-b8/summary.md"
  echo "  results/qwen25-tuning-tau1-b8/summary.csv"
} 2>&1 | tee "${LOG_FILE}"

echo "Console report log: ${LOG_FILE}"
