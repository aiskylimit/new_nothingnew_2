#!/usr/bin/env bash
# ALG-hard: ALG's soft weights against a hard keep/drop mask chosen by the same answer gain.
# DeepSeek-R1-Distill-Qwen-1.5B, P-ALIGN's 966 traces, E7's recipe (code: src/sgl/hardsel/).
#   bash project_commands_hard.sh                 # all: prepare, train A, nll_posthoc, report, train B, report
#   bash project_commands_hard.sh prepare         # CPU: step selection, checks, statistics, decision rule
#   bash project_commands_hard.sh train A         # H59 and R59, one per GPU   (B: H30, ALG-H59)
#   bash project_commands_hard.sh nll_posthoc     # NLL by region of the uniform / ALG checkpoints
#   bash project_commands_hard.sh report          # tables, figures, decision.json from disk
#   bash project_commands_hard.sh --dry-run       # print every command, run nothing
#   GPUS="0 1"  BUDGET=auto|fixed|sgl  HARD_GROUPS="A B"  GRADER=either|mathverify  KEEP_EPOCH_CKPT=0
#   HARD_ARMS="h59"  (train only these of h59 r59 h30 alg-h59)  NLL_SEEDS="42"  (nll_posthoc: seeds 43/44 need W4 group C checkpoints; default "42 43 44")
#   SEED42_UNIFORM=<results tag>  SEED42_ALG=<results tag>  SGL_ROW="..."  MODEL_PATH=...
# Needs the answer gain: experiments/answer_gain/scores/answer_gain.json, else W4's P0/S0 scoring
# (bash project_commands_w4.sh score). Finished steps are skipped: after a crash launch it again.
# Output: results/hard/ and data/<track>/train-hard-*.jsonl only.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

export GPUS="${GPUS:-0 1}"   # one job per GPU at a time
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
DATA_DIR="data/${TRACK}"
ROOT=results/hard
W4_ROOT="${W4_ROOT:-results/w4}"
TRACES=references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz
REF_GAINS="${REF_GAINS:-experiments/answer_gain/scores/answer_gain.json}"   # the gains ALG (E7) trained on
W4_SIGNALS="${W4_ROOT}/signals/r1q1.5b__P0__S0"                             # fallback: W4's re-scoring
REF_DATASET="${DATA_DIR}/train-e7-alg-tau1.jsonl"
SGL_MASK="${SGL_MASK:-${DATA_DIR}/train-spectral.jsonl}"                    # SGL gate masks, if kept
BUDGET="${BUDGET:-auto}"
SEED42_UNIFORM="${SEED42_UNIFORM:-e0-uniform-${TRACK}}"
SEED42_ALG="${SEED42_ALG:-e7-alg-tau1-${TRACK}}"
GRADER="${GRADER:-either}"
KEEP_EPOCH_CKPT="${KEEP_EPOCH_CKPT:-1}"   # 0: delete epoch1/epoch2 checkpoints once the run is evaluated
HARD_GROUPS="${HARD_GROUPS:-A B}"
EVAL_SEED=42
HASHES="${ROOT}/meta/hashes.json"

MODE=all DRY_RUN=0 PRIO=()
for arg in "$@"; do
  case "${arg}" in
    all|prepare|train|nll_posthoc|report) MODE="${arg}" ;;
    A|B) PRIO+=("${arg}") ;;   # group C (H59 seeds 43, 44) was dropped from the plan
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg} (modes: all prepare train nll_posthoc report; groups: A B; --dry-run)" >&2; exit 2 ;;
  esac
done
if [[ "${MODE}" == train && "${#PRIO[@]}" -eq 0 ]]; then PRIO=(A); fi

say() { echo "[$(date '+%F %T')] $*"; }
run() { if [[ "${DRY_RUN}" == 1 ]]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
run_id() { echo "r1q1.5b__$1__P0__S0__seed$2"; }   # <arm> <seed>

# run_queue <function> <job...>: one worker per GPU claims jobs in order (mkdir is atomic); every job
# runs to the end even if another fails; returns non-zero if any failed
run_queue() {
  local fn="$1"; shift
  local jobs=("$@") claims gpu i failed=0
  if [[ "${DRY_RUN}" == 1 ]]; then
    for i in "${!jobs[@]}"; do "${fn}" "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" ${jobs[$i]}; done
    return 0
  fi
  claims="$(mktemp -d)"
  worker() {
    local gpu="$1" i
    set +e   # a failed job is recorded below, it must not end the worker
    for i in "${!jobs[@]}"; do
      mkdir "${claims}/${i}" 2>/dev/null || continue
      ( set -e; "${fn}" "${gpu}" ${jobs[$i]} ) 2>&1 | sed -u "s/^/[gpu${gpu}] /"
      [[ "${PIPESTATUS[0]}" == 0 ]] || { echo "${jobs[$i]}" >> "${claims}/failed"; say "FAILED on gpu${gpu}: ${jobs[$i]}"; }
    done
  }
  for gpu in "${GPU_LIST[@]}"; do worker "${gpu}" & done
  wait
  if [[ -f "${claims}/failed" ]]; then failed=1; say "failed jobs: $(tr '\n' ';' < "${claims}/failed")"; fi
  rm -rf "${claims}"
  return "${failed}"
}

# ============================ CHECK ============================
if [[ "${DRY_RUN}" == 0 ]]; then
  mkdir -p "${ROOT}/meta"
  exec > >(tee -a "${ROOT}/meta/driver-$(date +%Y%m%d-%H%M%S).log") 2>&1
  say "mode=${MODE} groups=${PRIO[*]:-${HARD_GROUPS}} commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) GPUS=${GPUS}"
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"
  python -c "import transformers, pandas, pyarrow, matplotlib, math_verify, sgl; print('transformers', transformers.__version__)"
  if [[ "${MODE}" != report ]]; then
    [[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model: ${MODEL_PATH}" >&2; exit 1; }
    [[ -f "${TRACES}" ]] || { echo "missing training traces: ${TRACES}" >&2; exit 1; }
  fi
  if [[ "${MODE}" == all || "${MODE}" == train ]]; then
    for path in "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
      [[ -d "${path}" ]] || { echo "missing dataset: ${path}" >&2; exit 1; }
    done
    CUDA_VISIBLE_DEVICES="${GPUS// /,}" python -c "import torch, vllm, deepspeed; assert torch.cuda.is_available()"
    nvidia-smi -i "${GPUS// /,}" --query-gpu=index,name,memory.total,memory.used --format=csv
  fi
  for tag in "${SEED42_UNIFORM}" "${SEED42_ALG}"; do
    [[ -d "results/${tag}/raw" ]] || echo "WARNING: results/${tag}/raw is missing: no paired comparison with that seed-42 run" >&2
  done
fi

# ===================== STEP 1: SELECTION (CPU) =====================
prepare() {
  run python -m sgl.hardsel.report rule --root "${ROOT}" --grader "${GRADER}"
  if [[ -f "${ROOT}/selection/alg-h59/manifest.json" ]]; then
    say "selection: done already (delete ${ROOT}/selection to redo it)"; return 0
  fi
  say "data (train-segmented.jsonl) and input hashes"
  run python -m sgl.cli run "${TRACK}/e7-alg-tau1" --stages prepare "run.gpus=[${GPU_LIST[0]}]" "model.name=${MODEL_PATH}" \
    $([[ "${DRY_RUN}" == 1 ]] && echo --dry-run)
  run python -m sgl.robustness.runs hashes --traces "${TRACES}" --model-dir "${MODEL_PATH}" --output "${HASHES}"
  say "step selection, checks and training files (nothing is written for training if a check fails)"
  run python -m sgl.hardsel.select --data-path "${DATA_DIR}/train-segmented.jsonl" --tokenizer "${MODEL_PATH}" \
    --data-dir "${DATA_DIR}" --root "${ROOT}" --gains "${REF_GAINS}" --w4-signals "${W4_SIGNALS}" \
    --reference-dataset "${REF_DATASET}" --sgl-mask "${SGL_MASK}" --budget "${BUDGET}"
}

# ===================== STEP 2: TRAINING + EVALUATION =====================
train_eval() {  # <gpu> <arm> <seed>
  local gpu="$1" arm="$2" seed="$3" id dir config t0 t1 t2 overrides
  id="$(run_id "${arm}" "${seed}")"
  dir="${ROOT}/runs/${id}"
  config="${TRACK}/hard-${arm}"
  if [[ -f "${dir}/run_check.json" ]]; then say "${id}: done already"; return 0; fi
  # every step ends with "|| return 1": bash ignores set -e when a caller uses "||", and a failed run
  # must not get a run_check.json (it would count as done on the next launch)
  overrides=("train_seed=${seed}" "eval_seed=${EVAL_SEED}" "run.gpus=[${gpu}]" "model.name=${MODEL_PATH}")
  say "gpu${gpu}: ${id}"
  run mkdir -p "${dir}/train"
  if [[ "${DRY_RUN}" == 1 ]]; then
    run python -m sgl.cli show "${config}" "${overrides[@]}"
    python -m sgl.cli run "${config}" --stages train,eval "${overrides[@]}" --dry-run
  else
    python -m sgl.cli show "${config}" "${overrides[@]}" > "${dir}/config.yaml" || return 1
    t0=$(date +%s)
    python -m sgl.cli run "${config}" --stages train "${overrides[@]}" \
      > >(tee -a "${dir}/train/stdout.log") 2> >(tee -a "${dir}/train/stderr.log" >&2) || return 1
    t1=$(date +%s)
    rm -rf "${dir}"/checkpoints/epoch3/checkpoint-*   # optimizer state, only needed to resume
    python -m sgl.cli run "${config}" --stages eval "${overrides[@]}" > >(tee -a "${dir}/eval.log") 2>&1 || return 1
    t2=$(date +%s)
    printf '{"train_seconds": %d, "eval_seconds": %d, "finished_at": "%s"}\n' $((t1 - t0)) $((t2 - t1)) \
      "$(date -Iseconds)" > "${dir}/timing.json"
  fi
  # samples.jsonl + metrics.json (same columns as the W4 runs), then manifest + checklist
  run python -m sgl.robustness.runs eval --run-dir "${dir}" --config "${config}" --eval-seed "${EVAL_SEED}" || return 1
  run python -m sgl.hardsel.runs --run-dir "${dir}" --selection-root "${ROOT}/selection" --hashes "${HASHES}" \
    --eval-seed "${EVAL_SEED}" || return 1
  [[ "${KEEP_EPOCH_CKPT}" == 1 ]] || run rm -rf "${dir}/checkpoints/epoch1" "${dir}/checkpoints/epoch2"
}

train_group() {  # <A|B>
  local jobs=() entry arm seed file
  case "$1" in
    A) jobs=("h59 42" "r59 42") ;;
    B) jobs=("h30 42" "alg-h59 42") ;;
  esac
  if [[ -n "${HARD_ARMS:-}" ]]; then   # HARD_ARMS="h59": train only these arms of the group
    local kept=()
    for entry in "${jobs[@]}"; do [[ " ${HARD_ARMS} " == *" ${entry%% *} "* ]] && kept+=("${entry}"); done
    jobs=(${kept[@]+"${kept[@]}"})
    [[ "${#jobs[@]}" -gt 0 ]] || { say "group $1: no arm in HARD_ARMS=${HARD_ARMS}, nothing to train"; return 0; }
  fi
  for entry in "${jobs[@]}"; do   # every training file must exist before any GPU is taken
    read -r arm seed <<< "${entry}"
    file="${DATA_DIR}/train-hard-${arm}.jsonl"
    [[ "${DRY_RUN}" == 1 || -f "${file}" ]] || { echo "missing ${file}: run 'bash project_commands_hard.sh prepare' first" >&2; return 1; }
  done
  say "training group $1: ${#jobs[@]} runs, one per GPU"
  run_queue train_eval "${jobs[@]}"
}

# ===================== NLL of the uniform / ALG checkpoints =====================
nll_job() {  # <gpu> <method> <seed>
  local gpu="$1" method="$2" seed="$3" checkpoints=() probe=P0 seg=S0 w4 tag epoch
  if [[ "${seed}" == 42 ]]; then   # E0 / E7: only the final checkpoint exists
    tag="${SEED42_UNIFORM}"; [[ "${method}" == alg ]] && tag="${SEED42_ALG}"
    checkpoints=(--checkpoint "3=checkpoints/${tag}")
  else                             # W4 group C: every epoch
    [[ "${method}" == uniform ]] && { probe=none; seg=none; }
    w4="${W4_ROOT}/runs/r1q1.5b__${method}__${probe}__${seg}__tau1_lam0.5_c2__seed${seed}/checkpoints"
    for epoch in 1 2 3; do checkpoints+=(--checkpoint "${epoch}=${w4}/epoch${epoch}"); done
  fi
  checkpoints+=(--checkpoint "0=${MODEL_PATH}")   # before training: the base student
  say "gpu${gpu}: NLL of ${method} seed ${seed}"
  run env CUDA_VISIBLE_DEVICES="${gpu}" python -m sgl.hardsel.nll_eval --data-path "${DATA_DIR}/train-hard-h59.jsonl" \
    --tag "${method}__seed${seed}" --output "${ROOT}/posthoc_nll/${method}__seed${seed}/epoch_nll.json" "${checkpoints[@]}"
}
nll_posthoc() {
  [[ "${DRY_RUN}" == 1 || -f "${DATA_DIR}/train-hard-h59.jsonl" ]] || { echo "run prepare first" >&2; return 1; }
  local jobs=() method seed
  for method in uniform alg; do for seed in ${NLL_SEEDS:-42 43 44}; do jobs+=("${method} ${seed}"); done; done
  run_queue nll_job "${jobs[@]}"
}

report() {
  run python -m sgl.hardsel.report report --root "${ROOT}" --w4-root "${W4_ROOT}" --grader "${GRADER}" \
    --uniform-run "results/${SEED42_UNIFORM}" --alg-run "results/${SEED42_ALG}" ${SGL_ROW:+--sgl-row "${SGL_ROW}"}
}

case "${MODE}" in
  prepare) prepare ;;
  nll_posthoc) nll_posthoc ;;
  report) report ;;
  train)
    status=0
    for group in "${PRIO[@]}"; do train_group "${group}" || status=1; done
    [[ "${status}" == 0 ]] || { echo "some runs failed, see ${ROOT}/meta/driver-*.log and ${ROOT}/runs/<id>/" >&2; exit 1; } ;;
  all)
    prepare
    status=0
    [[ " ${HARD_GROUPS} " == *" A "* ]] && { train_group A || status=1; }
    nll_posthoc || status=1
    report || status=1
    if [[ " ${HARD_GROUPS} " == *" B "* ]]; then
      train_group B || status=1
      report || status=1
    fi
    [[ "${status}" == 0 ]] || { echo "some steps failed, see ${ROOT}/meta/driver-*.log" >&2; exit 1; } ;;
esac
say "done: ${ROOT}/"
