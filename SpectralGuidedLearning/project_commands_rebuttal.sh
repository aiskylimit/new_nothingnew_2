#!/usr/bin/env bash
# Rebuttal driver: answer-gain allocation on s1K-1.1 DeepSeek-R1 traces -- check, TRAIN, EVAL, compare.
# Models: r1-qwen-1.5b (full fine-tuning), qwen25-7b and qwen3-8b (LoRA r16). Configs: configs/sgl/<model>-s1k/.
# Arm: IWC-Stable weights from answer gain, lambda 0.5, no spectral gate; steps split by sentence.
# Just run it; it activates the venv, checks inputs and environment, then splits the runs over GPUs 0 and 1:
#   cd SpectralGuidedLearning && bash project_commands_rebuttal.sh
#   GPUS=3 bash project_commands_rebuttal.sh                 # other GPUs (space-separated ids)
#   MODELS=r1-qwen-1.5b ARMS=alg bash project_commands_rebuttal.sh   # some students / arms only
#   bash project_commands_rebuttal.sh --dry-run              # print every stage command, check and run nothing
# A rerun skips stages whose output exists (--force redoes them). Extra arguments go to every `sgl run`.
# Comment out any line you don't want to run.
# Outputs under rebuttal/: results/ (scores, comparison table), logs/ (one driver-<time>.log per launch with
# everything printed, plus <run>-<stage>.log of the latest run of each stage), answer-steps/<model>-s1k/
# (statistics and HTML view of the answer steps), efficiency/ (time and memory per stage, table.md). Datasets stay in data/, checkpoints in checkpoints/.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Runs are split over these GPUs, one GPU per run (training is NOT distributed; effective batch stays 32).
export GPUS="${GPUS:-0 1}"

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
DATASET_NAME="${DATASET_NAME:-${LOCAL_DATA_ROOT}/s1K-1.1}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
MODEL_DIRS=(DeepSeek-R1-Distill-Qwen-1.5B Qwen2.5-7B-Instruct Qwen3-8B)
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1

# ============================ CHECK ============================
if [[ "${DRY_RUN}" == 0 ]]; then
  # the full console output of this launch, kept across reruns (stage logs are overwritten)
  mkdir -p rebuttal/logs
  exec > >(tee -a "rebuttal/logs/driver-$(date +%Y%m%d-%H%M%S).log") 2>&1
  echo "[$(date '+%F %T')] commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), GPUS=${GPUS}, args: $*"
  echo "venv: ${PROJECT_ENV}  models: ${LOCAL_MODELS_ROOT}  data: ${LOCAL_DATA_ROOT}"

  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"

  echo "--- ${LOCAL_DATA_ROOT}"; ls "${LOCAL_DATA_ROOT}" || true
  echo "--- ${LOCAL_MODELS_ROOT}"; ls "${LOCAL_MODELS_ROOT}" || true
  # every missing input is listed before stopping, so one launch shows all of them
  missing=0
  for path in "${DATASET_NAME}" "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
    [[ -d "${path}" ]] || { echo "missing dataset: ${path} (see download.txt)" >&2; missing=1; }
  done
  for model in "${MODEL_DIRS[@]}"; do
    [[ -f "${LOCAL_MODELS_ROOT}/${model}/config.json" ]] || {
      echo "missing model: ${LOCAL_MODELS_ROOT}/${model} (see download.txt)" >&2; missing=1; }
  done
  [[ "${missing}" == 0 ]] || exit 1

  nvidia-smi -i "${GPUS// /,}" --query-gpu=index,name,memory.total,memory.used --format=csv
  # fail now rather than hours in if the venv lacks part of the train or eval stack
  CUDA_VISIBLE_DEVICES="${GPUS// /,}" python -c "
import sys, torch, transformers, peft, deepspeed, vllm, datasets, math_verify, pandas, matplotlib, sgl
print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__,
      'vllm', vllm.__version__)
assert torch.cuda.is_available(), 'CUDA is not available'
"
fi

TRAIN_STAGES=prepare,capture,answer_gain,gain_signal,weights,answer_stats,train

# run_model <model> <gpu>: everything for one student, one stage after the other
run_model() {
  local model="$1" gpu="$2" dir lora=-lora
  case "${model}" in
    r1-qwen-1.5b) dir=DeepSeek-R1-Distill-Qwen-1.5B; lora="" ;;
    qwen25-7b) dir=Qwen2.5-7B-Instruct ;;
    qwen3-8b) dir=Qwen3-8B ;;
  esac
  # answer gain: data -> capture -> gain -> signal -> weights -> answer-step statistics -> train
  has_arm alg && sgl_run "${gpu}" "${model}-s1k/iwc-gain-l05${lora}" "${dir}" "${TRAIN_STAGES}" "${ARGS[@]}"
  # baseline: plain SFT on every step (NLL)
  has_arm sft && sgl_run "${gpu}" "${model}-s1k/sft-nll${lora}" "${dir}" prepare,vanilla,train "${ARGS[@]}"
  # eval: answer gain, NLL, zero-shot
  has_arm alg && sgl_run "${gpu}" "${model}-s1k/iwc-gain-l05${lora}" "${dir}" eval "${ARGS[@]}"
  has_arm sft && sgl_run "${gpu}" "${model}-s1k/sft-nll${lora}" "${dir}" eval "${ARGS[@]}"
  has_arm zeroshot && sgl_run "${gpu}" "${model}-s1k/zeroshot" "${dir}" eval "${ARGS[@]}"
  return 0   # a skipped last arm must not count as a failure
}

# One run trains on one GPU; the GPUs work side by side. All arms of a model stay on one GPU, in order,
# because they share data/<model>-s1k/. Each free GPU takes the next model of the list.
read -ra GPU_LIST <<< "${GPUS//,/ }"
# MODELS="qwen3-8b" pins a launch to some students (several launches on disjoint GPUs: disjoint data dirs)
read -ra MODELS <<< "${MODELS:-r1-qwen-1.5b qwen25-7b qwen3-8b}"
# ARMS="alg" keeps only some arms of each model: alg (answer gain), sft (NLL), zeroshot
read -ra ARM_LIST <<< "${ARMS:-alg sft zeroshot}"
has_arm() { [[ " ${ARM_LIST[*]} " == *" $1 "* ]]; }
ARGS=("$@")

# sgl_run <gpu> <config> <model dir name> <stages> [extra args]
sgl_run() {
  python -m sgl.cli run "$2" --stages "$4" "run.gpus=[$1]" \
    "model.name=${LOCAL_MODELS_ROOT}/$3" "dataset=${DATASET_NAME}" "${@:5}"
}

# worker <gpu>: claim models one at a time (mkdir is atomic); a failed model does not stop the others
worker() {
  local gpu="$1" model status
  for model in "${MODELS[@]}"; do
    mkdir "${CLAIMS}/${model}" 2>/dev/null || continue
    echo "[$(date '+%F %T')] gpu${gpu}: start ${model}"
    set +e
    ( set -e; run_model "${model}" "${gpu}" ) 2>&1 | sed -u "s/^/[gpu${gpu} ${model}] /"
    status=${PIPESTATUS[0]}
    set -e
    echo "[$(date '+%F %T')] gpu${gpu}: ${model} finished, exit ${status}"
    [[ "${status}" == 0 ]] || echo "${model}" >> "${CLAIMS}/failed"
  done
}

FAILED=""
if [[ "${DRY_RUN}" == 1 ]]; then
  for i in "${!MODELS[@]}"; do run_model "${MODELS[$i]}" "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"; done
else
  CLAIMS="$(mktemp -d)"
  trap 'rm -rf "${CLAIMS}"' EXIT
  trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops the background workers too
  for gpu in "${GPU_LIST[@]}"; do worker "${gpu}" & done
  wait
  [[ -f "${CLAIMS}/failed" ]] && FAILED="$(tr '\n' ' ' < "${CLAIMS}/failed")"
fi

# =========================== COMPARE ==========================
# writes rebuttal/results/comparison-table.md and efficiency/table.md from whatever finished
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.eval.compare --results-dir rebuttal/results
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.diagnostics.efficiency
[[ -z "${FAILED}" ]] || { echo "FAILED models: ${FAILED}(see rebuttal/logs/)" >&2; exit 1; }
