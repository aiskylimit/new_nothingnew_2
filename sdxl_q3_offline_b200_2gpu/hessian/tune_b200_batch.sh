#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT
source "$PROJECT_ROOT/env.sh"

TARGET_VRAM_PERCENT="${TARGET_VRAM_PERCENT:-95}"
AUTOTUNE_STEPS="${AUTOTUNE_STEPS:-10}"
BATCH_CANDIDATES="${B200_BATCH_CANDIDATES:-32 40 48 56 60 64 68 72 80}"
STATUS_LOG="$RUNTIME_ROOT/B200_AUTOTUNE_STATUS.log"
SELECTED_ENV="$RUNTIME_ROOT/b200-autotune.env"
mkdir -p "$RUNTIME_ROOT/logs" "$RUNS_DIR"

status() {
  printf '%s | %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_LOG"
}

IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
(( ${#gpu_array[@]} == NUM_GPUS )) || {
  status "failed invalid_gpu_count ids=$GPU_IDS num_gpus=$NUM_GPUS"
  exit 10
}

inventory="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits)"
status "inventory $(printf '%s' "$inventory" | tr '\n' ';')"
for gpu in "${gpu_array[@]}"; do
  row="$(printf '%s\n' "$inventory" | awk -F, -v id="$gpu" '$1+0 == id {print; exit}')"
  [[ -n "$row" ]] || { status "failed missing_gpu=$gpu"; exit 11; }
  name="$(printf '%s' "$row" | cut -d, -f2)"
  used="$(printf '%s' "$row" | cut -d, -f3 | tr -d ' ')"
  [[ "$name" == *B200* ]] || { status "failed gpu=$gpu is_not_B200 name=$name"; exit 12; }
  if (( used > 2048 )) && [[ "${AUTOTUNE_ALLOW_BUSY:-0}" != 1 ]]; then
    status "failed gpu=$gpu already_uses_${used}MiB"
    exit 13
  fi
done

selected_batch=""
selected_effective=""
selected_peak=""
for micro_batch in $BATCH_CANDIDATES; do
  effective_batch=$((NUM_GPUS * micro_batch))
  run_name="b200_vram_tune_mb${micro_batch}_eb${effective_batch}"
  run_dir="$RUNS_DIR/$run_name"
  monitor="$RUNTIME_ROOT/logs/${run_name}-gpu.csv"
  rm -rf -- "$run_dir"
  : > "$monitor"

  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits >> "$monitor"
      sleep 0.2
    done
  ) &
  monitor_pid=$!

  status "candidate_start micro_batch=$micro_batch effective_batch=$effective_batch"
  set +e
  GPU_IDS="$GPU_IDS" NUM_GPUS="$NUM_GPUS" TRAIN_BATCH_SIZE="$micro_batch" \
    EFFECTIVE_BATCH="$effective_batch" DATASET_PAIRS=4096 \
    PILOT_TRAIN_STEPS="$AUTOTUNE_STEPS" RUN_NAME="$run_name" \
    DISABLE_CPU_OFFLOAD=1 SKIP_FINAL_SAVE=1 \
    CHECKPOINTING_STEPS=999999 CHECKPOINTS_TOTAL_LIMIT=1 \
    bash "$PROJECT_ROOT/hessian/run_offline_train.sh" pilot \
    > "$RUNTIME_ROOT/logs/${run_name}.log" 2>&1
  rc=$?
  set -e
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  peak_percent="$($VENV_DIR/bin/python - "$monitor" "$GPU_IDS" <<'PY'
import csv
import sys

path, gpu_ids = sys.argv[1], {int(x) for x in sys.argv[2].split(',')}
peak = 0.0
with open(path, newline='', encoding='utf-8') as handle:
    for row in csv.reader(handle):
        if len(row) < 4:
            continue
        try:
            gpu, used, total = int(row[1]), float(row[2]), float(row[3])
        except ValueError:
            continue
        if gpu in gpu_ids and total > 0:
            peak = max(peak, 100.0 * used / total)
print(f"{peak:.2f}")
PY
)"
  status "candidate_end micro_batch=$micro_batch rc=$rc peak_percent=$peak_percent"
  rm -rf -- "$run_dir"

  if (( rc != 0 )); then
    break
  fi
  if ! awk -v peak="$peak_percent" -v target="$TARGET_VRAM_PERCENT" \
      'BEGIN { exit !(peak <= target) }'; then
    break
  fi
  selected_batch="$micro_batch"
  selected_effective="$effective_batch"
  selected_peak="$peak_percent"
done

[[ -n "$selected_batch" ]] || {
  status "failed no_candidate_below_${TARGET_VRAM_PERCENT}_percent"
  exit 20
}

full_steps=$(((851293 + selected_effective - 1) / selected_effective))
cat > "$SELECTED_ENV" <<EOF
export TRAIN_BATCH_SIZE=$selected_batch
export EFFECTIVE_BATCH=$selected_effective
export MAX_TRAIN_STEPS=$full_steps
export DISABLE_CPU_OFFLOAD=1
export B200_TUNED_PEAK_PERCENT=$selected_peak
EOF
status "selected micro_batch=$selected_batch effective_batch=$selected_effective full_steps=$full_steps peak_percent=$selected_peak env=$SELECTED_ENV"
