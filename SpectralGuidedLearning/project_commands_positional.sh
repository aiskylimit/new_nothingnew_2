#!/usr/bin/env bash
# Is answer gain only a position effect? DeepSeek-R1-Distill-Qwen-1.5B, 966 traces (c=2, tau=1, lambda=0.5).
#   bash project_commands_positional.sh               # everything: analyses, R6, decision, R7/R8 if the rule asks
#   bash project_commands_positional.sh analyze       # tables, weights, sanity checks, A1-A4 only (minutes, CPU)
#   bash project_commands_positional.sh report        # rebuild report.md from what is on disk
#   bash project_commands_positional.sh --dry-run     # print the commands, run nothing
#   RUNS_EXTRA=r7 bash project_commands_positional.sh # also run R7 (the plan's "A1b profiles differ" case)
#   FOLLOWUPS=0 bash project_commands_positional.sh   # R6 and the decision only: never R7/R8
#   PALIGN_RUN=<tag> ALERD_RUN=<tag> ...              # which results/<tag> are the published baselines
# Everything is written under results/positional/ (report.md, decision.json, analysis/, runs/<id>/).
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

export GPUS="${GPUS:-0 1}"   # one run per GPU
read -ra GPU_LIST <<< "${GPUS//,/ }"

PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_PATH="${MODEL_PATH:-${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

TRACK=r1-qwen-1.5b-palign
ALERD_CONFIG="${TRACK}/e7-alg-tau1"                       # ALERD as published: tau=1, lambda=0.5, clip=2
DATA_DIR="data/${TRACK}"
GAINS="${GAINS:-experiments/answer_gain/scores/answer_gain.json}"
LOGP="${LOGP:-experiments/answer_gain/scores/answer_logp.json}"
ALERD_RUN="${ALERD_RUN:-e7-alg-tau1-${TRACK}}"
PALIGN_RUN="${PALIGN_RUN:-}"                              # empty: the run in results/ with 40.76 / 52.03
ROOT=results/positional

MODE=all DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    all|analyze|report) MODE="${arg}" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

# run <command...>: print it in dry-run mode, execute it otherwise
run() { if [[ "${DRY_RUN}" == 1 ]]; then echo "+ $*"; else "$@"; fi; }
sgl_run() {  # <gpu> <config> <stages> [overrides]
  local flags=()
  [[ "${DRY_RUN}" == 1 ]] && flags=(--dry-run)
  python -m sgl.cli run "$2" --stages "$3" "run.gpus=[$1]" "model.name=${MODEL_PATH}" "${@:4}" "${flags[@]}"
}
say() { echo "[$(date '+%F %T')] $*"; }

# ============================ CHECK ============================
if [[ "${DRY_RUN}" == 0 ]]; then
  mkdir -p "${ROOT}/meta"
  exec > >(tee -a "${ROOT}/meta/driver-$(date +%Y%m%d-%H%M%S).log") 2>&1
  say "mode=${MODE} commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) GPUS=${GPUS}"
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"
  [[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model: ${MODEL_PATH}" >&2; exit 1; }
  python -c "import transformers, pandas, pyarrow, matplotlib, math_verify, sgl; print('transformers', transformers.__version__)"
  if [[ "${MODE}" == all ]]; then
    for path in "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
      [[ -d "${path}" ]] || { echo "missing dataset: ${path}" >&2; exit 1; }
    done
    CUDA_VISIBLE_DEVICES="${GPUS// /,}" python -c "import torch, vllm, deepspeed; assert torch.cuda.is_available()"
    nvidia-smi -i "${GPUS// /,}" --query-gpu=index,name,memory.total,memory.used --format=csv
  fi
fi

if [[ "${MODE}" == report ]]; then
  run python -m sgl.positional.decide --root "${ROOT}" --report-only
  exit 0
fi

# ========================= STEP 1: ANALYSIS =========================
say "data and answer gains"
sgl_run "${GPU_LIST[0]}" "${ALERD_CONFIG}" prepare
if [[ ! -f "${GAINS}" ]]; then
  echo "WARNING: ${GAINS} not found, scoring answer gain again (GPU). These may not be bit-identical to the" >&2
  echo "         scores ALERD was trained on; check analysis/sanity_weights.json against Table 8." >&2
  sgl_run "${GPU_LIST[0]}" "${ALERD_CONFIG}" answer_gain "stages.answer_gain.args.logp-output=${LOGP}"
fi

TABLE_ARGS=(--data-path "${DATA_DIR}/train-segmented.jsonl" --gains "${GAINS}" --tokenizer "${MODEL_PATH}"
            --output-dir "${ROOT}" --logp "${LOGP}" --seed 42)
# if ALERD's own training file is still there, the recomputed weights must match it exactly
[[ -f "${DATA_DIR}/train-e7-alg-tau1.jsonl" ]] && TABLE_ARGS+=(--alerd-dataset "${DATA_DIR}/train-e7-alg-tau1.jsonl")
say "tables, R6/R7/R8 weights, sanity checks (training does not start if one fails)"
run python -m sgl.positional.tables "${TABLE_ARGS[@]}"
say "analyses A1-A4"
run python -m sgl.positional.analysis --root "${ROOT}" --seed 42
[[ "${MODE}" == analyze ]] && { say "done: ${ROOT}/analysis/"; exit 0; }

# =================== BASELINES (raw generations) ====================
# paired bootstrap needs per-sample outputs of P-ALIGN and ALERD: reuse results/<tag>/raw, else re-evaluate
# the checkpoint with the same protocol
BASELINES=()
baseline() {  # <name> <run tag>
  local name="$1" tag="$2" raw
  if [[ -z "${tag}" ]]; then
    echo "WARNING: no run tag for ${name} (set PALIGN_RUN / ALERD_RUN): no paired bootstrap against it" >&2
    return 0
  fi
  if [[ -f "results/${tag}/raw/aime25.jsonl" ]]; then
    raw="results/${tag}/raw"
  elif [[ -f "checkpoints/${tag}/config.json" ]]; then
    say "re-evaluating ${name} from checkpoints/${tag}"
    sgl_run "${GPU_LIST[0]}" "${ALERD_CONFIG}" eval "results_dir=${ROOT}/runs" "run_name=${name}" \
      "checkpoint=checkpoints/${tag}"
    raw="${ROOT}/runs/${name}/raw"
  else
    echo "WARNING: ${name}: neither results/${tag}/raw nor checkpoints/${tag}: no paired bootstrap" >&2
    return 0
  fi
  run python -m sgl.positional.evalreport report --raw-dir "${raw}" --out-dir "${ROOT}/runs/${name}/eval" \
    --config "${ALERD_CONFIG}"
  BASELINES+=(--baseline "${name}=${ROOT}/runs/${name}/eval")
}
if [[ -z "${PALIGN_RUN}" && "${DRY_RUN}" == 0 ]]; then
  PALIGN_RUN="$(python -m sgl.positional.evalreport resolve --avg1 40.76 --avg3 52.03 --results-dir results)"
  say "P-ALIGN run with 40.76 / 52.03: ${PALIGN_RUN:-not found}"
fi
baseline palign "${PALIGN_RUN}"
baseline alerd "${ALERD_RUN}"

# ========================= STEP 2: THE RUNS =========================
train_eval() {  # <gpu> <r6|r7|r8>
  local gpu="$1" id="$2"
  say "gpu${gpu}: train + eval ${id}"
  sgl_run "${gpu}" "${TRACK}/pos-${id}" train,eval
  run python -m sgl.positional.evalreport report --raw-dir "${ROOT}/runs/${id}/raw" \
    --out-dir "${ROOT}/runs/${id}/eval" --config "${TRACK}/pos-${id}" "${BASELINES[@]}"
}
train_eval "${GPU_LIST[0]}" r6

# ========================= STEP 3: DECISION =========================
run python -m sgl.positional.decide --root "${ROOT}"
NEXT=""
[[ "${DRY_RUN}" == 0 ]] && NEXT="$(python -m sgl.positional.decide --root "${ROOT}" --print-next)"
read -ra MORE <<< "$(printf '%s\n' ${NEXT} ${RUNS_EXTRA:-} | sort -u | tr '\n' ' ')"
if [[ "${FOLLOWUPS:-1}" == 0 && "${#MORE[@]}" -gt 0 ]]; then
  say "FOLLOWUPS=0: not running ${MORE[*]} (the rule's verdict is still in ${ROOT}/decision.json)"
  MORE=()
fi
if [[ "${#MORE[@]}" -gt 0 ]]; then
  say "follow-up runs: ${MORE[*]}"
  pids=() failed=0
  for i in "${!MORE[@]}"; do
    train_eval "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" "${MORE[$i]}" &
    pids+=($!)
    # at most one run per GPU at a time
    if (( (i + 1) % ${#GPU_LIST[@]} == 0 || i + 1 == ${#MORE[@]} )); then
      for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
      pids=()
    fi
  done
  [[ "${failed}" == 0 ]] || { echo "a follow-up run failed, see ${ROOT}/meta/logs/" >&2; exit 1; }
  run python -m sgl.positional.decide --root "${ROOT}" --report-only
fi
say "done: ${ROOT}/report.md and ${ROOT}/decision.json"
