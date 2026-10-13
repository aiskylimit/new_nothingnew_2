#!/bin/bash
# GRIT (GRPO safety + NSPO null-space projection + KL trust region) on Qwen2.5-3B-Instruct, 1 x B200 180 GB.
# Order: check env -> verify prepared artifacts -> dry run (3 steps) -> train (40 steps)
#        -> eval base + checkpoints (safety + general) -> summary table.
# Layout: colocated, world size 1 - FP32 actor + FP32 vLLM rollout + bf16 Qwen3Guard on the same GPU.
# Every stage is resumable: re-running skips finished work (verified artifacts, dry-run marker,
# training resumes from the latest checkpoint, eval resumes per model/benchmark).
# Memory: the dry run tries MEM_CONFIGS from most to least aggressive and keeps the first that passes
#   (vLLM sleep mode frees its GPU memory for the actor's backward/curvature phases).
# Eval: EVAL_SLOTS vLLM engines per GPU run models in parallel (base, step 20, step 40 at once).
# commands.sh runs exactly "cd ./GRIT" + "bash ./project_commands.sh"; to change a setting, edit its
# default below (GPU, RUN_ID, LAMBDA_PRES, MAX_STEPS, STAGES, EVAL_STEPS, EVAL_MAX_SAMPLES, MEM_CONFIGS,
# MEM_CONFIG, EVAL_GPUS, EVAL_SLOTS). The same names also work as env vars when running by hand.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH=.

VENV=/mnt/local/uvenvs/grit
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "ERROR: ${VENV} does not exist. Run the env job first: header '#i grit.txt' in commands.sh" >&2
  exit 1
fi
source "${VENV}/bin/activate"

# Paths - must match download.txt.
PROJECT_NAME="aiskylimit_new_nothingnew_2"
BASE_DIR="/mnt/local/${PROJECT_NAME}/grit"
MODEL="${BASE_DIR}/models/Qwen2.5-3B-Instruct"
GUARD="${BASE_DIR}/models/Qwen3Guard-Gen-4B"
ARTIFACTS="${BASE_DIR}/prepared/Qwen--Qwen2.5-3B-Instruct"
RAW="${BASE_DIR}/data/raw"
TASK_FILE="${ARTIFACTS}/data/task/task_train.parquet"
EVAL_FILE="${ARTIFACTS}/data/eval/evaluation_prompts.parquet"
PRESERVE_FILE="${ARTIFACTS}/data/preservation/split_v1/qwen--qwen2.5-3b-instruct/monitor/preserve_contexts.parquet"
PROJECTORS="${ARTIFACTS}/artifacts/qwen--qwen2.5-3b-instruct_split_v1_projectors.pt"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# commands.sh only runs "cd ./GRIT && bash ./project_commands.sh", so every default below must be right
# as is. GPU: the free-est GPU is picked automatically (the server is shared); set a number to pin it.
GPU=${GPU:-auto}
if [ "${GPU}" == "auto" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  echo "auto-selected GPU ${GPU} (${FREE_MB} MiB free)"
  if [ "${FREE_MB}" -lt 150000 ]; then
    echo "ERROR: no GPU has >= 150 GB free (best: GPU ${GPU}, ${FREE_MB} MiB); GRIT needs one whole B200" >&2
    nvidia-smi
    exit 1
  fi
fi
RUN_ID=${RUN_ID:-grit_3b_lam1}
LAMBDA_PRES=${LAMBDA_PRES:-1}
MAX_STEPS=${MAX_STEPS:-40}
STAGES=${STAGES:-check,verify,dryrun,train,eval,summary}
# Checkpoints are saved every 5 steps; only these are evaluated (base is always evaluated).
EVAL_STEPS=${EVAL_STEPS:-"20 40"}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-}   # e.g. 50 for a quick eval smoke test
EVAL_GPUS=${EVAL_GPUS:-${GPU}}           # e.g. "0,1" if more GPUs are free for eval
EVAL_SLOTS=${EVAL_SLOTS:-3}              # concurrent vLLM engines per eval GPU
# Memory configs, TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP, most aggressive first.
# The last one is the original PLAN.md section 3 setting (no sleep mode) as a safe fallback.
MEM_CONFIGS=${MEM_CONFIGS:-"16:4:64:0.40:1 8:2:32:0.40:1 8:2:32:0.30:0"}
OUT_DIR="${BASE_DIR}/outputs/${RUN_ID}"  # checkpoints (~12 GB each) stay on /mnt/local
EVAL_WORK="${BASE_DIR}/eval/${RUN_ID}"   # generations, guard labels
RESULTS="${ROOT}/results_${RUN_ID}"      # small files only: logs, metrics, eval results, summary
mkdir -p "${RESULTS}/logs" "${RESULTS}/train" "${RESULTS}/eval" "${BASE_DIR}/outputs"
# Everything printed below also goes to one log per run (console output is unchanged).
exec > >(tee -a "${RESULTS}/logs/full_run_$(date +%Y%m%d_%H%M%S).log") 2>&1
for path in "${MODEL}/config.json" "${GUARD}/config.json" "${ARTIFACTS}/SHA256SUMS.json"; do
  [ -f "${path}" ] || { echo "ERROR: missing ${path}. Run the download job first: GRIT/download.txt as a '#d' job"; exit 1; }
done

T0=$(date +%s)
on_exit() {  # always show where things stand, also after a failure or a timeout kill
  local code=$?
  if [ "${code}" != "0" ]; then
    echo; echo "=== FAILED/STOPPED (exit ${code}) after $(( $(date +%s) - T0 ))s - partial summary follows; logs: ${RESULTS}/logs ==="
    copy_train_results "${OUT_DIR}" "${RESULTS}/train" || true
    python scripts/summarize_results.py --results-dir "${RESULTS}" --title "GRIT ${RUN_ID} (partial)" || true
  fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT  # a timeout kill still runs on_exit with a non-zero code
stage() { echo; echo "=== [$(( $(date +%s) - T0 ))s] $* ==="; }
want() { [[ ",${STAGES}," == *",$1,"* ]]; }

# Fixed 3B configuration (PLAN.md section 3); memory knobs come from the selected MEM_CONFIG.
TRAIN_ARGS=(
  --model-path "${MODEL}" --safety-model-path "${GUARD}"
  --task-file "${TASK_FILE}" --preserve-file "${PRESERVE_FILE}" --evaluation-file "${EVAL_FILE}"
  --projectors-path "${PROJECTORS}" --projector-relaxation 0.05
  --rollout-gpu "${GPU}" --allow-colocated-rollout
  --rollout-dtype float32
  --task-batch-size 321 --preserve-batch-size 48
  --generations 5 --max-prompt-length 512 --max-response-length 256 --max-preserve-length 2304
  --task-optimizer adamw --lr 1e-6 --weight-decay 0 --adam-eps 1e-8
  --lambda-pres "${LAMBDA_PRES}" --epsilon-pres 0.05 --top-k 64 --base-statistics-dtype float32
  --use-curvature --fd-radius 0.05 --fd-check-interval 10
  --gradient-checkpointing --seed 66
)

mem_args() {  # mem_args TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP -> trainer flags
  local task pres reward mem sleep
  IFS=: read -r task pres reward mem sleep <<< "$1"
  MEM_ARGS=(--task-microbatch-size "${task}" --preservation-microbatch-size "${pres}"
            --reward-batch-size "${reward}" --rollout-memory-utilization "${mem}")
  [ "${sleep}" == "1" ] && MEM_ARGS+=(--rollout-sleep)
  return 0
}

latest_checkpoint() {  # newest complete checkpoint that can still be resumed
  local dir
  dir=$(ls -1d "$1"/step_* 2>/dev/null | sort | while read -r d; do
          [ -f "$d/complete.json" ] && [ -f "$d/training.pt" ] && echo "$d"; done | tail -1 || true)
  echo "${dir}"
}

copy_train_results() {  # small training outputs into the project folder
  local src=$1 dst=$2
  [ -d "${src}" ] || return 0
  mkdir -p "${dst}"
  for f in metrics.jsonl diagnostics_summary.json; do [ -f "${src}/${f}" ] && cp "${src}/${f}" "${dst}/"; done
  [ -d "${src}/validation" ] && cp -r "${src}/validation" "${dst}/"
  return 0
}

train() {  # train <output_dir> <max_steps> <save_steps> <validation_steps> <log>
  local out=$1 steps=$2 save=$3 validation=$4 log=$5 resume=() latest
  latest=$(latest_checkpoint "${out}")
  if [ -n "${latest}" ]; then
    if [ "$(basename "${latest}")" == "$(printf 'step_%06d' "${steps}")" ]; then
      echo "skip: ${out} already reached step ${steps}"; return 0
    fi
    echo "resume from ${latest}"; resume=(--resume "${latest}")
  elif [ -d "${out}" ]; then
    # Crashed before the first checkpoint: keep it for inspection, start clean.
    mv "${out}" "${out}.failed.$(date +%Y%m%d_%H%M%S)"
  fi
  env CUDA_VISIBLE_DEVICES="${GPU}" python scripts/train_grit.py "${TRAIN_ARGS[@]}" "${MEM_ARGS[@]}" \
      --output-dir "${out}" --max-steps "${steps}" --save-steps "${save}" --validation-steps "${validation}" \
      --prune-old-training-state ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "${log}"
}

# ============================================================
# 1. Environment check (fail fast) + CPU regression tests
# ============================================================
if want check; then
  stage "1. environment check"
  nvidia-smi -i "${GPU}" || true
  env CUDA_VISIBLE_DEVICES="${GPU}" python - <<'PYEOF' 2>&1 | tee "${RESULTS}/logs/check.log"
import torch, transformers, vllm, datasets
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}, "
      f"vllm {vllm.__version__}, datasets {datasets.__version__}")
assert torch.cuda.is_available(), "no CUDA device visible"
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {name}, capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}, "
      f"free {free/1e9:.0f}/{total/1e9:.0f} GB")
if free < 150e9:
    print("WARNING: less than 150 GB free on this GPU; the colocated 3B layout needs ~145-165 GB")
PYEOF
  python -m pytest -q -p no:cacheprovider 2>&1 | tail -3 | tee -a "${RESULTS}/logs/check.log"
fi

# ============================================================
# 2. Verify prepared artifacts (replaces the prepare stage; see PLAN.md 4.2.1)
# ============================================================
if want verify; then
  stage "2. verify artifacts"
  env CUDA_VISIBLE_DEVICES="${GPU}" python scripts/verify_artifacts.py \
      --artifact-root "${ARTIFACTS}" --model-path "${MODEL}" --logit-check 8 \
      2>&1 | tee "${RESULTS}/logs/verify.log"
fi

# ============================================================
# 3. Dry run: 3 steps incl. one validation and one checkpoint, to surface path/VRAM/env errors cheaply
# ============================================================
if want dryrun; then
  stage "3. dry run (3 steps) + memory config selection"
  if [ -f "${RESULTS}/dryrun.ok" ]; then
    echo "skip: dry run already passed with MEM_CONFIG=$(cat "${RESULTS}/memory_config.txt" 2>/dev/null)"
  else
    chosen=""
    for config in ${MEM_CONFIGS}; do
      echo "--- dry run with MEM_CONFIG=${config} (TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP)"
      mem_args "${config}"
      rm -rf "${OUT_DIR}_dryrun"
      if train "${OUT_DIR}_dryrun" 3 3 2 "${RESULTS}/logs/dryrun.log"; then chosen="${config}"; break; fi
      echo "dry run FAILED with ${config} (see ${RESULTS}/logs/dryrun.log); trying the next config"
    done
    [ -n "${chosen}" ] || { echo "dry run failed with every MEM_CONFIG - see ${RESULTS}/logs/dryrun.log"; exit 1; }
    copy_train_results "${OUT_DIR}_dryrun" "${RESULTS}/dryrun"
    rm -rf "${OUT_DIR}_dryrun"   # ~37 GB checkpoint, not needed after the check
    echo "${chosen}" > "${RESULTS}/memory_config.txt"
    date > "${RESULTS}/dryrun.ok"
    echo "Dry run passed with MEM_CONFIG=${chosen} - per step:"
    python -c "import json;[print(' step',r['step'],f\"peak {r['peak_vram_gb']:.0f}/180 GB\",f\"{r['step_seconds']:.0f}s\") for r in map(json.loads,open('${RESULTS}/dryrun/metrics.jsonl'))]"
  fi
fi

# ============================================================
# 4. Train (resumes from the latest checkpoint; checkpoints every 5 steps, validation every 5)
# ============================================================
if want train; then
  # Explicit MEM_CONFIG > the one the dry run selected > the safe original setting.
  MEM_CONFIG=${MEM_CONFIG:-$(cat "${RESULTS}/memory_config.txt" 2>/dev/null || echo "8:2:32:0.30:0")}
  # On failure (e.g. OOM on a longer KL context than the dry run saw), fall back to the next, less
  # aggressive config and resume from the latest checkpoint; no partial step is ever committed.
  fallbacks="${MEM_CONFIG}"; seen=0
  for config in ${MEM_CONFIGS}; do
    [ "${seen}" == "1" ] && fallbacks="${fallbacks} ${config}"
    [ "${config}" == "${MEM_CONFIG}" ] && seen=1
  done
  trained=0
  for config in ${fallbacks}; do
    mem_args "${config}"
    stage "4. train ${RUN_ID}: lambda_pres=${LAMBDA_PRES}, ${MAX_STEPS} steps, MEM_CONFIG=${config}"
    if train "${OUT_DIR}" "${MAX_STEPS}" 5 5 "${RESULTS}/logs/train.log"; then trained=1; break; fi
    echo "train FAILED with MEM_CONFIG=${config}; falling back to the next config"
    echo "${config} failed $(date)" >> "${RESULTS}/memory_config_failures.txt"
  done
  copy_train_results "${OUT_DIR}" "${RESULTS}/train"
  [ "${trained}" == "1" ] || { echo "training failed with every MEM_CONFIG - see ${RESULTS}/logs/train.log"; exit 1; }
fi

# ============================================================
# 5. Eval: base + selected checkpoints, PKU safety (1,000) + general benchmarks + AlpacaEval over-refusal
# ============================================================
if want eval; then
  stage "5. eval (base + steps: ${EVAL_STEPS})"
  copy_train_results "${OUT_DIR}" "${RESULTS}/train"
  MODELS=(--model "base=${MODEL}")
  for step in ${EVAL_STEPS}; do
    ckpt="${OUT_DIR}/$(printf 'step_%06d' "${step}")"
    if [ -f "${ckpt}/complete.json" ]; then MODELS+=(--model "step_$(printf '%06d' "${step}")=${ckpt}")
    else echo "skip (no complete checkpoint): ${ckpt}"; fi
  done
  python scripts/eval_run.py "${MODELS[@]}" --raw-root "${RAW}" --safety-file "${EVAL_FILE}" \
      --guard-path "${GUARD}" --work-dir "${EVAL_WORK}" --results-dir "${RESULTS}/eval" \
      --gpus "${EVAL_GPUS}" --slots-per-gpu "${EVAL_SLOTS}" \
      ${EVAL_MAX_SAMPLES:+--max-samples "${EVAL_MAX_SAMPLES}"} 2>&1 | tee -a "${RESULTS}/logs/eval.log"
fi

# ============================================================
# 6. Summary: training curve + eval table + paired deltas -> console, summary.md, summary.json
# ============================================================
if want summary; then
  stage "6. summary"
  copy_train_results "${OUT_DIR}" "${RESULTS}/train"
  python scripts/summarize_results.py --results-dir "${RESULTS}" --title "GRIT ${RUN_ID}"
fi

echo "ALL DONE - $(( $(date +%s) - T0 ))s. Results: ${RESULTS}/summary.md"
