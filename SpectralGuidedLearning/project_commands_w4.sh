#!/usr/bin/env bash
# W4 (reviewer HHj8): is answer-gain allocation robust to the probe and the step segmentation?
# DeepSeek-R1-Distill-Qwen-1.5B, P-ALIGN's 966 traces, full FT, tau=1, lambda=0.5, c=2 (code: src/sgl/robustness/).
#   bash project_commands_w4.sh                 # score + rank stability, then train/eval A, B, C, then the tables
#   bash project_commands_w4.sh score           # scoring + signals + rank stability only (no training)
#   bash project_commands_w4.sh train A         # one or more priority groups: A (P1, Pavg, S1, S3),
#                                               #   B (P2, P3, S2a, S2b), C (uniform + ALG default, seeds 43/44)
#   bash project_commands_w4.sh report          # rebuild results_master.csv and the two tables from disk
#   bash project_commands_w4.sh --dry-run       # print every command, run nothing
#   GPUS="0 1"  KEEP_EPOCH_CKPT=0  SEED42_UNIFORM=<results tag>  SEED42_ALG=<results tag>  MODEL_PATH=...
# Finished steps are skipped, so after a crash launch the same command again.
# Everything lands under results/w4/ (signals/, runs/, analysis/, meta/); training files in data/<track>/.
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
CONFIG="${TRACK}/w4"
DATA_DIR="data/${TRACK}"
SEG_DIR="${DATA_DIR}/w4"
ROOT=results/w4
TRACES=references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz
REF_GAINS="${REF_GAINS:-experiments/answer_gain/scores/answer_gain.json}"   # the gains ALERD trained on
REF_DATASET="${DATA_DIR}/train-e7-alg-tau1.jsonl"                          # ALERD's training file, if kept
SEED42_UNIFORM="${SEED42_UNIFORM:-e0-uniform-${TRACK}}"    # seed 42 of the noise runs = the existing runs
SEED42_ALG="${SEED42_ALG:-e7-alg-tau1-${TRACK}}"
KEEP_EPOCH_CKPT="${KEEP_EPOCH_CKPT:-1}"   # 0: delete epoch1/epoch2 checkpoints once the run is evaluated
EVAL_SEED=42
HASHES="${ROOT}/meta/hashes.json"
SEGS=(S0 S1 S2a S2b S3)
SCORED=("P0 S0" "P1 S0" "P2 S0" "P3 S0" "P0 S1" "P0 S2a" "P0 S2b" "P0 S3")
declare -A PROBE_TEXT=([P0]=$'\n\nTherefore, the final answer is \\boxed{' [P1]=$'\n\nThe answer is \\boxed{'
                       [P2]=$'\n\nFinal Answer: \\boxed{' [P3]=$'\n\n**Final Answer**\n\n\\boxed{')

MODE=all DRY_RUN=0 PRIO=()
for arg in "$@"; do
  case "${arg}" in
    all|score|train|report) MODE="${arg}" ;;
    A|B|C) PRIO+=("${arg}") ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg} (modes: all score train report; groups: A B C; --dry-run)" >&2; exit 2 ;;
  esac
done
[[ "${#PRIO[@]}" -gt 0 ]] || PRIO=(A B C)

say() { echo "[$(date '+%F %T')] $*"; }
run() { if [[ "${DRY_RUN}" == 1 ]]; then printf '+'; printf ' %q' "$@"; echo; else "$@"; fi; }
signal_dir() { echo "${ROOT}/signals/r1q1.5b__$1__$2"; }
run_id() {  # <method> <probe> <seg> <seed>
  if [[ "$1" == uniform ]]; then echo "r1q1.5b__uniform__none__none__tau1_lam0.5_c2__seed$4"
  else echo "r1q1.5b__$1__$2__$3__tau1_lam0.5_c2__seed$4"; fi
}

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
  say "mode=${MODE} groups=${PRIO[*]} commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown) GPUS=${GPUS}"
  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"
  [[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model: ${MODEL_PATH}" >&2; exit 1; }
  [[ -f "${TRACES}" ]] || { echo "missing training traces: ${TRACES}" >&2; exit 1; }
  python -c "import transformers, pandas, pyarrow, math_verify, sgl; print('transformers', transformers.__version__)"
  if [[ "${MODE}" != report ]]; then
    for path in "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
      [[ -d "${path}" ]] || { echo "missing dataset: ${path}" >&2; exit 1; }
    done
    CUDA_VISIBLE_DEVICES="${GPUS// /,}" python -c "import torch, vllm, deepspeed; assert torch.cuda.is_available()"
    nvidia-smi -i "${GPUS// /,}" --query-gpu=index,name,memory.total,memory.used --format=csv
  fi
fi

report() {
  say "rank stability (Table R-W4a) and results_master.csv (Table R-W4b)"
  run python -m sgl.robustness.stability --root "${ROOT}" --seed 42
  run python -m sgl.robustness.runs master --root "${ROOT}" \
    --seed42 "uniform=results/${SEED42_UNIFORM}" --seed42 "alg=results/${SEED42_ALG}"
  say "tables: ${ROOT}/analysis/table_r_w4a.md, ${ROOT}/analysis/table_r_w4b.md"
}
if [[ "${MODE}" == report ]]; then report; exit 0; fi

# ===================== STEP 1: SCORING (no training) =====================
score_job() {  # <gpu> <probe> <seg>   answer gain of one probe on one segmentation
  local gpu="$1" probe="$2" seg="$3" dir sampler cmd
  dir="$(signal_dir "${probe}" "${seg}")/raw"
  if [[ -f "${dir}/gains.json" ]]; then say "score ${probe}/${seg}: done already"; return 0; fi
  say "gpu${gpu}: score ${probe}/${seg}"
  cmd=(python -m sgl.signals.answer_gain --data-path "${SEG_DIR}/segmented-${seg}.jsonl" --model "${MODEL_PATH}"
       --probe "${PROBE_TEXT[${probe}]}" --output "${dir}/gains.json" --logp-output "${dir}/logp.json"
       --meta-output "${dir}/meta.json")
  if [[ "${DRY_RUN}" == 1 ]]; then run env CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"; return 0; fi
  mkdir -p "${dir}"
  # peak GPU memory while vLLM scores (MiB, sampled every second)
  nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "${dir}/gpu_mem.txt" &
  sampler=$!
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > >(tee "${dir}/score.log") 2>&1 || { kill "${sampler}"; return 1; }
  kill "${sampler}" 2>/dev/null || true
}
logp_job() {  # <gpu> tokenlogp   frozen-student log-prob of every response token (predictability column)
  if [[ -f "${ROOT}/meta/token_logp.npz" ]]; then say "token log-probs: done already"; return 0; fi
  say "gpu$1: token log-probs of the frozen student"
  run env CUDA_VISIBLE_DEVICES="$1" python -m sgl.robustness.token_logp --data-path "${SEG_DIR}/segmented-S0.jsonl" \
    --model "${MODEL_PATH}" --output "${ROOT}/meta/token_logp.npz"
}
job() { if [[ "$2" == tokenlogp ]]; then logp_job "$@"; else score_job "$@"; fi; }

signals() {  # <probe> <seg>   tables, weights, checks and the training file of one variant
  local probe="$1" seg="$2" out raws=() args=()
  out="$(signal_dir "${probe}" "${seg}")"
  if [[ "${probe}" == Pavg ]]; then raws=(P0 P1 P2 P3); else raws=("${probe}"); fi
  args=(--data-path "${SEG_DIR}/segmented-${seg}.jsonl" --probe "${probe}" --seg "${seg}" --tokenizer "${MODEL_PATH}"
        --out-dir "${out}" --dataset-output "${DATA_DIR}/train-w4-${probe}-${seg}.jsonl" --hashes "${HASHES}"
        --token-logp "${ROOT}/meta/token_logp.npz" --gains)
  for p in "${raws[@]}"; do args+=("$(signal_dir "${p}" "${seg}")/raw/gains.json"); done
  args+=(--logp); for p in "${raws[@]}"; do args+=("$(signal_dir "${p}" "${seg}")/raw/logp.json"); done
  args+=(--meta); for p in "${raws[@]}"; do args+=("$(signal_dir "${p}" "${seg}")/raw/meta.json"); done
  [[ "${probe}" == Pavg ]] || args+=(--gpu-mem-file "${out}/raw/gpu_mem.txt")
  if [[ "${probe}/${seg}" == P0/S0 ]]; then
    # regression against the published scores and weights; also the uniform training file (same tags)
    args+=(--reference-gains "${REF_GAINS}" --uniform-output "${DATA_DIR}/train-w4-uniform.jsonl")
    [[ -f "${REF_DATASET}" ]] && args+=(--reference-dataset "${REF_DATASET}")
  fi
  run python -m sgl.robustness.signals "${args[@]}"
}

if [[ "${MODE}" == all || "${MODE}" == score ]]; then
  say "data (S0 = train-segmented.jsonl) and input hashes"
  run python -m sgl.cli run "${TRACK}/e7-alg-tau1" --stages prepare "run.gpus=[${GPU_LIST[0]}]" "model.name=${MODEL_PATH}" \
    $([[ "${DRY_RUN}" == 1 ]] && echo --dry-run)
  run python -m sgl.robustness.runs hashes --traces "${TRACES}" --model-dir "${MODEL_PATH}" --output "${HASHES}"
  for seg in "${SEGS[@]}"; do
    [[ -f "${SEG_DIR}/segmented-${seg}.jsonl" ]] && continue
    say "segmentation ${seg} (coverage checks; S0 must reproduce train-segmented.jsonl exactly)"
    run python -m sgl.robustness.segment --data-path "${DATA_DIR}/train-segmented.jsonl" --seg "${seg}" \
      --tokenizer "${MODEL_PATH}" --output "${SEG_DIR}/segmented-${seg}.jsonl"
  done
  [[ -f "${REF_GAINS}" ]] || echo "WARNING: ${REF_GAINS} not found: P0/S0 is not checked against the published gains" >&2
  say "scoring: token log-probs + ${#SCORED[@]} probe/segmentation pairs over GPUs ${GPUS}"
  run_queue job tokenlogp "${SCORED[@]}"
  say "signals, weights and training files (training does not start if a check fails)"
  for pair in "${SCORED[@]}" "Pavg S0"; do signals ${pair}; done
  report
  [[ "${MODE}" == score ]] && { say "done: ${ROOT}/analysis/table_r_w4a.md"; exit 0; }
fi

# ===================== STEP 2: TRAINING + EVALUATION =====================
train_eval() {  # <gpu> <method> <probe> <seg> <seed>
  local gpu="$1" method="$2" probe="$3" seg="$4" seed="$5" id dir variant signals_dir t0 t1 t2 overrides
  id="$(run_id "${method}" "${probe}" "${seg}" "${seed}")"
  dir="${ROOT}/runs/${id}"
  if [[ "${method}" == uniform ]]; then variant=w4-uniform; signals_dir=""
  else variant="w4-${probe}-${seg}"; signals_dir="$(signal_dir "${probe}" "${seg}")"; fi
  if [[ -f "${dir}/run_check.json" ]]; then say "${id}: done already"; return 0; fi
  # every step ends with "|| return 1": bash ignores set -e when a caller uses "||", and a failed run
  # must not get a run_check.json (it would count as done on the next launch)
  overrides=("run_name=${id}" "train_variant=${variant}" "train_seed=${seed}" "eval_seed=${EVAL_SEED}"
             "run.gpus=[${gpu}]" "model.name=${MODEL_PATH}")
  say "gpu${gpu}: ${id}"
  run mkdir -p "${dir}/train"
  if [[ "${DRY_RUN}" == 1 ]]; then
    run python -m sgl.cli show "${CONFIG}" "${overrides[@]}"
    python -m sgl.cli run "${CONFIG}" --stages train,eval "${overrides[@]}" --dry-run
  else
    python -m sgl.cli show "${CONFIG}" "${overrides[@]}" > "${dir}/config.yaml" || return 1
    t0=$(date +%s)
    python -m sgl.cli run "${CONFIG}" --stages train "${overrides[@]}" \
      > >(tee -a "${dir}/train/stdout.log") 2> >(tee -a "${dir}/train/stderr.log" >&2) || return 1
    t1=$(date +%s)
    rm -rf "${dir}"/checkpoints/epoch3/checkpoint-*   # optimizer state, only needed to resume
    python -m sgl.cli run "${CONFIG}" --stages eval "${overrides[@]}" > >(tee -a "${dir}/eval.log") 2>&1 || return 1
    t2=$(date +%s)
    printf '{"train_seconds": %d, "eval_seconds": %d, "finished_at": "%s"}\n' $((t1 - t0)) $((t2 - t1)) \
      "$(date -Iseconds)" > "${dir}/timing.json"
  fi
  run python -m sgl.robustness.runs eval --run-dir "${dir}" --config "${CONFIG}" --eval-seed "${EVAL_SEED}" || return 1
  run python -m sgl.robustness.runs finalize --run-dir "${dir}" --hashes "${HASHES}" --eval-seed "${EVAL_SEED}" \
    ${signals_dir:+--signals-dir "${signals_dir}"} || return 1
  [[ "${KEEP_EPOCH_CKPT}" == 1 ]] || run rm -rf "${dir}/checkpoints/epoch1" "${dir}/checkpoints/epoch2"
}

if [[ "${MODE}" == all || "${MODE}" == train ]]; then
  RUNS=()
  for group in "${PRIO[@]}"; do
    case "${group}" in
      A) RUNS+=("alg P1 S0 42" "alg Pavg S0 42" "alg P0 S1 42" "alg P0 S3 42") ;;
      B) RUNS+=("alg P2 S0 42" "alg P3 S0 42" "alg P0 S2a 42" "alg P0 S2b 42") ;;
      C) for seed in 43 44; do RUNS+=("uniform none none ${seed}" "alg P0 S0 ${seed}"); done ;;
    esac
  done
  for entry in "${RUNS[@]}"; do   # every training file must exist (step 1) before any GPU is taken
    read -r method probe seg _ <<< "${entry}"
    file="${DATA_DIR}/train-w4-${probe}-${seg}.jsonl"; [[ "${method}" == uniform ]] && file="${DATA_DIR}/train-w4-uniform.jsonl"
    [[ "${DRY_RUN}" == 1 || -f "${file}" ]] || { echo "missing ${file}: run 'bash project_commands_w4.sh score' first" >&2; exit 1; }
  done
  say "training ${#RUNS[@]} runs (groups ${PRIO[*]}), one per GPU"
  status=0
  run_queue train_eval "${RUNS[@]}" || status=1
  report
  [[ "${status}" == 0 ]] || { echo "some runs failed, see ${ROOT}/meta/driver-*.log and ${ROOT}/runs/<id>/" >&2; exit 1; }
fi
say "done: ${ROOT}/analysis/"
