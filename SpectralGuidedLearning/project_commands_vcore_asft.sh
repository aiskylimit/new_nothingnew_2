#!/usr/bin/env bash
# VCORE and ASFT baselines: DeepSeek-R1-Distill-Qwen-1.5B on P-ALIGN's 966 traces, same data, schedule and
# eval as the other r1-qwen-1.5b-palign arms. Port and its differences from upstream: src/sgl/training/baselines.py.
#   bash project_commands_vcore_asft.sh                         # ASFT and VCORE, one per GPU (GPUs 0,1)
#   GPUS=0 bash project_commands_vcore_asft.sh                  # one after the other on GPU 0
#   ARMS=sft-vcore bash project_commands_vcore_asft.sh          # one arm only
#   bash project_commands_vcore_asft.sh asft_kl_lambda=0.03     # any sgl override is passed through
#   bash project_commands_vcore_asft.sh --dry-run               # print every stage command, run nothing
# Outputs: rebuttal/results/sft-{asft,vcore}-r1-qwen-1.5b-palign/, rebuttal/logs/, rebuttal/efficiency/.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Each arm trains on ONE GPU (VCORE cannot run distributed); effective batch stays 32.
export GPUS="${GPUS:-0 1}"
read -ra GPU_LIST <<< "${GPUS//,/ }"
read -ra ARM_LIST <<< "${ARMS:-sft-asft sft-vcore}"
TRACK=r1-qwen-1.5b-palign

# Offline server: no HF Hub access, load from local mirrors (see download.txt).
PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/mnt/local/_data/aiskylimit_new_nothingnew_2}"
export BENCH_DATA_ROOT="${BENCH_DATA_ROOT:-${LOCAL_DATA_ROOT}}"
LOCAL_MODELS_ROOT="${LOCAL_MODELS_ROOT:-/mnt/local/_models/aiskylimit_new_nothingnew_2}"
MODEL_PATH="${LOCAL_MODELS_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 DO_NOT_TRACK=1 FLASHINFER_NO_DOWNLOAD=1
export WANDB_DISABLED=true WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false

DRY_RUN=0
[[ " $* " == *" --dry-run "* ]] && DRY_RUN=1
ARGS=("$@")

# ============================ CHECK ============================
if [[ "${DRY_RUN}" == 0 ]]; then
  # the full console output of this launch, kept across reruns (stage logs are overwritten)
  mkdir -p rebuttal/logs
  exec > >(tee -a "rebuttal/logs/driver-vcore-asft-$(date +%Y%m%d-%H%M%S).log") 2>&1
  echo "[$(date '+%F %T')] commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), GPUS=${GPUS}, args: $*"
  echo "venv: ${PROJECT_ENV}  model: ${MODEL_PATH}  benchmarks: ${BENCH_DATA_ROOT}"

  [[ -f "${PROJECT_ENV}/bin/activate" ]] || { echo "venv not found: ${PROJECT_ENV} (set PROJECT_ENV)" >&2; exit 1; }
  source "${PROJECT_ENV}/bin/activate"

  missing=0
  for path in "${BENCH_DATA_ROOT}"/{AIME_2024,aime_2025,MATH-500,aimo-validation-amc}; do
    [[ -d "${path}" ]] || { echo "missing dataset: ${path} (see download.txt)" >&2; missing=1; }
  done
  [[ -f "${MODEL_PATH}/config.json" ]] || { echo "missing model: ${MODEL_PATH} (see download.txt)" >&2; missing=1; }
  traces=references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz
  [[ -f "${traces}" ]] || { echo "missing training traces: ${traces}" >&2; missing=1; }
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

# sgl_run <gpu> <arm> <stages>
sgl_run() {
  python -m sgl.cli run "${TRACK}/$2" --stages "$3" "run.gpus=[$1]" "model.name=${MODEL_PATH}" "${ARGS[@]}"
}

# ============================= DATA ============================
# shared by every arm of the track: build it once, before the arms start side by side
sgl_run "${GPU_LIST[0]}" "${ARM_LIST[0]}" prepare,vanilla

# ========================= TRAIN + EVAL ========================
# worker <gpu>: claim arms one at a time (mkdir is atomic); a failed arm does not stop the others
worker() {
  local gpu="$1" arm status
  for arm in "${ARM_LIST[@]}"; do
    mkdir "${CLAIMS}/${arm}" 2>/dev/null || continue
    echo "[$(date '+%F %T')] gpu${gpu}: start ${arm}"
    set +e
    ( set -e; sgl_run "${gpu}" "${arm}" train; sgl_run "${gpu}" "${arm}" eval ) 2>&1 \
      | sed -u "s/^/[gpu${gpu} ${arm}] /"
    status=${PIPESTATUS[0]}
    set -e
    echo "[$(date '+%F %T')] gpu${gpu}: ${arm} finished, exit ${status}"
    [[ "${status}" == 0 ]] || echo "${arm}" >> "${CLAIMS}/failed"
  done
}

FAILED=""
if [[ "${DRY_RUN}" == 1 ]]; then
  for i in "${!ARM_LIST[@]}"; do
    gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
    sgl_run "${gpu}" "${ARM_LIST[$i]}" train
    sgl_run "${gpu}" "${ARM_LIST[$i]}" eval
  done
else
  CLAIMS="$(mktemp -d)"
  trap 'rm -rf "${CLAIMS}"' EXIT
  trap 'trap - INT TERM; kill 0' INT TERM   # Ctrl-C stops the background workers too
  # wait for the workers by pid: a bare `wait` also waits for the `exec > >(tee ...)` log process
  # (some bash versions), which only ends with this script, so the driver would hang forever
  worker_pids=()
  for gpu in "${GPU_LIST[@]}"; do worker "${gpu}" & worker_pids+=($!); done
  wait "${worker_pids[@]}"
  [[ -f "${CLAIMS}/failed" ]] && FAILED="$(tr '\n' ' ' < "${CLAIMS}/failed")"
fi

# =========================== COMPARE ==========================
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.eval.compare --results-dir rebuttal/results
[[ "${DRY_RUN}" == 1 ]] || python -m sgl.diagnostics.efficiency
[[ -z "${FAILED}" ]] || { echo "FAILED arms: ${FAILED}(see rebuttal/logs/)" >&2; exit 1; }
