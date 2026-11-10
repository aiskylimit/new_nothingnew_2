#!/usr/bin/env bash
set -euo pipefail

# Usage: bash runs/eval_math_code.sh [--list|--train-sft] [BASE_PATH] [FAMILY] [METHOD] [VARIANT]
#        bash runs/eval_math_code.sh --sft-only [BASE_PATH] [FAMILY]
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST_ONLY=0
TRAIN_SFT=0
SFT_ONLY=0
case "${1:-}" in
    --list) LIST_ONLY=1; shift ;;
    --train-sft) TRAIN_SFT=1; shift ;;
    --sft-only) TRAIN_SFT=1; SFT_ONLY=1; shift ;;
esac

BASE_PATH="$(cd "${1:-${PROJECT_ROOT}}" && pwd)"
FAMILY=${2:-all}
METHOD=${3:-all}
VARIANT=${4:-all}
EVAL_BACKEND=${EVAL_BACKEND:-vllm}
TASKS=${TASKS:-gsm8k_cot,hendrycks_math,humaneval,mbpp}
OPENLLAMA2_BASE_MODEL=${OPENLLAMA2_BASE_MODEL:-"${BASE_PATH}/checkpoints/openllama2-3B"}
OPENLLAMA2_MERGED_ROOT=${OPENLLAMA2_MERGED_ROOT:-}
GPT2_CONTRA_DIR=${GPT2_CONTRA_DIR:-contra_sampled}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-"${BASE_PATH}/results/eval_math_code"}

fail() {
    echo "$*" >&2
    exit 1
}

has_model() {
    local path=$1
    [[ -f "${path}/config.json" ]] &&
        [[ -f "${path}/model.safetensors" || -f "${path}/model.safetensors.index.json" ||
           -f "${path}/pytorch_model.bin" || -f "${path}/pytorch_model.bin.index.json" ]]
}

require_model() {
    has_model "$1" || fail "Missing model config or weights in $1"
}

has_adapter() {
    local path=$1
    [[ -f "${path}/adapter_config.json" ]] &&
        [[ -f "${path}/adapter_model.safetensors" || -f "${path}/adapter_model.bin" ]]
}

if [[ ${SFT_ONLY} == 1 ]]; then
    (( $# <= 2 )) || fail "Usage: bash project_commands.sh --sft-only [BASE_PATH] [FAMILY]"
    METHOD=sft
    OPENLLAMA2_MERGED_ROOT=
fi

case "${FAMILY}" in all|gpt2|openllama2) ;; *) fail "Unknown family: ${FAMILY}" ;; esac
case "${METHOD}" in all|distillm|distillm2|csd|fdd|sft|pretrained) ;; *) fail "Unknown method: ${METHOD}" ;; esac
case "${VARIANT}" in all|baseline|contra) ;; *) fail "Unknown variant: ${VARIANT}" ;; esac
if [[ ${SFT_ONLY} == 0 ]]; then
    case "${EVAL_BACKEND}" in vllm|hf) ;; *) fail "Unknown evaluation backend: ${EVAL_BACKEND}" ;; esac
fi

CHECKPOINTS=()
FAMILIES=()
METHODS=()
while read -r family method variant checkpoint; do
    [[ ${FAMILY} == all || ${FAMILY} == "${family}" ]] || continue
    [[ ${METHOD} == all || ${METHOD} == "${method}" ]] || continue
    [[ ${VARIANT} == all || ${VARIANT} == "${variant}" ]] || continue
    if [[ ${family} == gpt2 && ${variant} == contra ]]; then
        checkpoint="${GPT2_CONTRA_DIR}/${checkpoint}"
    fi
    CHECKPOINTS+=("${checkpoint}")
    FAMILIES+=("${family}")
    METHODS+=("${method}")
done <<'CHECKPOINTS'
gpt2 distillm baseline distillm/0.1B_1.5B
gpt2 distillm contra distillm/0.1B_1.5B
gpt2 distillm2 baseline distillm2/v2_base
gpt2 distillm2 contra distillm2/0.1B_1.5B
gpt2 csd baseline csd/0.1B_1.5B
gpt2 csd contra csd/0.1B_1.5B
gpt2 fdd baseline fdd/base
gpt2 fdd contra fdd/base
openllama2 distillm baseline distillm/3B_7B
openllama2 distillm contra contra/distillm/3B_7B
openllama2 distillm2 baseline distillm2/3B_7B
openllama2 distillm2 contra contra/distillm2/3B_7B
openllama2 csd baseline csd/3B_7B
openllama2 csd contra contra/csd/3B_7B
openllama2 fdd baseline fdd/3B_7B
openllama2 fdd contra contra/fdd/3B_7B
gpt2 sft baseline sft/gpt2-base
openllama2 sft baseline sft/openllama2-3B
gpt2 pretrained baseline pretrained/gpt2-base
gpt2 pretrained baseline pretrained/gpt2-xlarge
openllama2 pretrained baseline pretrained/openllama2-3B
openllama2 pretrained baseline pretrained/openllama2-7B
CHECKPOINTS

(( ${#CHECKPOINTS[@]} > 0 )) || fail "No checkpoints match the requested family, method, and variant."

if [[ ${FAMILY} != gpt2 && -z ${OPENLLAMA2_MERGED_ROOT} ]]; then
    echo "OpenLLaMA2 base model: ${OPENLLAMA2_BASE_MODEL}"
    if [[ ${LIST_ONLY} == 0 ]]; then
        require_model "${OPENLLAMA2_BASE_MODEL}"
    fi
fi

MODEL_PATHS=()
SFT_INDICES=()
for i in "${!CHECKPOINTS[@]}"; do
    family=${FAMILIES[$i]}
    method=${METHODS[$i]}
    checkpoint=${CHECKPOINTS[$i]}
    case "${checkpoint}" in
        pretrained/openllama2-3B) model_path="${OPENLLAMA2_BASE_MODEL}" ;;
        pretrained/*) model_path="${BASE_PATH}/checkpoints/${checkpoint#pretrained/}" ;;
        *) model_path="${BASE_PATH}/results/${family}/train/${checkpoint}" ;;
    esac
    if [[ ${family} == openllama2 && ${method} != pretrained && -n ${OPENLLAMA2_MERGED_ROOT} ]]; then
        model_path="${OPENLLAMA2_MERGED_ROOT}/${checkpoint}"
    fi
    MODEL_PATHS+=("${model_path}")
    if [[ ${LIST_ONLY} == 1 ]]; then
        printf '%s\t%s\n' "${family}" "${model_path}"
    elif [[ ${TRAIN_SFT} == 1 && ${method} == sft && ( ${family} == gpt2 || -z ${OPENLLAMA2_MERGED_ROOT} ) ]]; then
        SFT_INDICES+=("${i}")
    elif [[ ${family} == openllama2 && ${method} != pretrained && -z ${OPENLLAMA2_MERGED_ROOT} ]]; then
        has_adapter "${model_path}" || fail "Missing LoRA config or weights in ${model_path}"
    else
        require_model "${model_path}"
    fi
done
if [[ ${LIST_ONLY} == 1 ]]; then
    exit 0
fi

ENV_DIR=${ENV_DIR:-/mnt/local/uvenvs/contra}
if [[ -f "${ENV_DIR}/bin/activate" && ${VIRTUAL_ENV:-} != "${ENV_DIR}" ]]; then
    source "${ENV_DIR}/bin/activate"
fi
if [[ ${SFT_ONLY} == 0 ]]; then
    command -v lm-eval >/dev/null || fail "Provision ../contra.txt and activate the contra environment."
    export HF_ALLOW_CODE_EVAL=1
fi

export HF_HOME=${HF_HOME:-"${BASE_PATH}/.cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

EVAL_ARGS=(--tasks "${TASKS}" --batch_size "${BATCH_SIZE:-auto}" --log_samples
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS:-256}" --confirm_run_unsafe_code)
if [[ -n ${LIMIT:-} ]]; then
    EVAL_ARGS+=(--limit "${LIMIT}")
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}/smoke"
fi

evaluate_checkpoint() {
    local i=$1
    local family=${FAMILIES[$i]}
    local method=${METHODS[$i]}
    local model_path=${MODEL_PATHS[$i]}
    local model_args output_dir status
    model_args="pretrained=${model_path},dtype=float16"
    output_dir="${EVAL_OUTPUT_DIR}/${EVAL_BACKEND}/${family}/${CHECKPOINTS[$i]}"
    if [[ ${family} == openllama2 && ${method} != pretrained && -z ${OPENLLAMA2_MERGED_ROOT} ]]; then
        model_args="pretrained=${OPENLLAMA2_BASE_MODEL},dtype=float16"
        if [[ -f "${model_path}/tokenizer.json" || -f "${model_path}/tokenizer.model" ]]; then
            model_args+=",tokenizer=${model_path}"
        fi
        if [[ ${EVAL_BACKEND} == vllm ]]; then
            model_args+=",lora_local_path=${model_path},max_lora_rank=${MAX_LORA_RANK:-16}"
        else
            model_args+=",peft=${model_path}"
        fi
    elif [[ ${family} == openllama2 && ${method} != pretrained ]]; then
        output_dir="${EVAL_OUTPUT_DIR}/${EVAL_BACKEND}/openllama2-merged/${CHECKPOINTS[$i]}"
    fi
    if [[ ${EVAL_BACKEND} == vllm ]]; then
        model_args+=",tensor_parallel_size=1"
    fi

    mkdir -p "${output_dir}" || return
    echo "[GPU ${CUDA_VISIBLE_DEVICES}] Evaluating ${model_path}. Log at ${output_dir}/eval.log"
    if lm-eval run --model "${EVAL_BACKEND}" --model_args "${model_args}" \
        "${EVAL_ARGS[@]}" --output_path "${output_dir}" > "${output_dir}/eval.log" 2>&1; then
        echo "[GPU ${CUDA_VISIBLE_DEVICES}] Completed ${model_path}"
    else
        status=$?
        echo "[GPU ${CUDA_VISIBLE_DEVICES}] Failed ${model_path} (exit ${status}). See ${output_dir}/eval.log" >&2
        return "${status}"
    fi
}

GPU_LIST=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
case "${GPU_LIST}" in
    ,*|*,|*,,*|*[[:space:]]*) fail "Expected comma-separated GPU IDs in CUDA_VISIBLE_DEVICES." ;;
esac
IFS=, read -r -a GPU_IDS <<< "${GPU_LIST}"
seen=,
for gpu in "${GPU_IDS[@]}"; do
    [[ ${seen} != *",${gpu},"* ]] || fail "Duplicate GPU in CUDA_VISIBLE_DEVICES: ${gpu}"
    seen+="${gpu},"
done

# The training scripts export tokenizer files to the checkpoint root after saving weights.
sft_ready() {
    local family=$1 path=$2
    [[ -f "${path}/tokenizer_config.json" ]] || return 1
    if [[ ${family} == openllama2 ]]; then
        has_adapter "${path}"
    else
        has_model "${path}"
    fi
}

train_student() {
    local i=$1 port=$2
    local family=${FAMILIES[$i]} model_path=${MODEL_PATHS[$i]}
    local script base_model
    case "${family}" in
        gpt2)
            script="scripts/gpt2/sft/sft_base.sh"
            base_model="${BASE_PATH}/checkpoints/gpt2-base"
            ;;
        openllama2)
            script="scripts/openllama2/sft/sft_3B_lora.sh"
            base_model="${OPENLLAMA2_BASE_MODEL}"
            ;;
    esac

    if sft_ready "${family}" "${model_path}"; then
        echo "Using existing student SFT checkpoint ${model_path}"
        return
    fi
    require_model "${base_model}"

    local processed_root="${BASE_PATH}/processed_data/dolly/full"
    local data_path="${processed_root}/${family}"
    local log_dir="${BASE_PATH}/logs/student_sft"
    local dolly_data_dir=${DOLLY_DATA_DIR:-"$(dirname "${BASE_PATH}")/contra-kd/data/raw/dolly"}
    mkdir -p "${log_dir}"
    if [[ ! -s "${data_path}/train_0.bin" || ! -s "${data_path}/train_0.idx" ||
          ! -s "${data_path}/valid_0.bin" || ! -s "${data_path}/valid_0.idx" ||
          ! -s "${data_path}/valid.jsonl" ]]; then
        [[ -s "${dolly_data_dir}/raw.jsonl" ]] || fail "Missing Dolly data at ${dolly_data_dir}/raw.jsonl"
        echo "Preparing Dolly for ${family}. Log at ${log_dir}/${family}_data.log"
        if ! CUDA_VISIBLE_DEVICES="" PYTHONPATH="${BASE_PATH}" python3 -u "${BASE_PATH}/tools/process_data_dolly.py" \
            --data-dir "${dolly_data_dir}" --processed-data-dir "${processed_root}" \
            --model-path "${base_model}" --model-type "${family}" --max-prompt-length 256 \
            --dev-num 1000 --data-process-workers "${DATA_PROCESS_WORKERS:-4}" \
            > "${log_dir}/${family}_data.log" 2>&1; then
            fail "Dolly preprocessing failed for ${family}. See ${log_dir}/${family}_data.log"
        fi
    fi

    echo "[GPU ${CUDA_VISIBLE_DEVICES}] Training ${family} student SFT. Log at ${log_dir}/${family}.log"
    if ! bash "${BASE_PATH}/${script}" "${BASE_PATH}" "${port}" 1 --model-path "${base_model}" \
        > "${log_dir}/${family}.log" 2>&1; then
        fail "Student SFT failed for ${family}. See ${log_dir}/${family}.log"
    fi
    sft_ready "${family}" "${model_path}" || fail "Student SFT did not export ${model_path}"
}

train_on_gpu() {
    local gpu_index=$1 i
    for ((i = gpu_index; i < ${#SFT_INDICES[@]}; i += ${#GPU_IDS[@]})); do
        train_student "${SFT_INDICES[$i]}" "$((${SFT_BASE_PORT:-2012} + i))"
    done
}

evaluate_on_gpu() {
    local gpu_index=$1 i status=0
    for ((i = gpu_index; i < ${#CHECKPOINTS[@]}; i += ${#GPU_IDS[@]})); do
        evaluate_checkpoint "${i}" || status=$?
    done
    return "${status}"
}

# Give each worker a process group so cleanup also stops its vLLM processes.
set -m
PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        [[ -n ${pid} ]] || continue
        kill -- "-${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_on_gpus() {
    local worker=$1 count=$2 i status=0
    for i in "${!GPU_IDS[@]}"; do
        if (( i >= count )); then
            break
        fi
        (
            set +m
            export CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}"
            "${worker}" "${i}"
        ) &
        PIDS+=("$!")
    done

    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || status=$?
        unset 'PIDS[i]'
    done
    return "${status}"
}

if (( ${#SFT_INDICES[@]} > 0 )); then
    run_on_gpus train_on_gpu "${#SFT_INDICES[@]}"
fi
if [[ ${SFT_ONLY} == 0 ]]; then
    run_on_gpus evaluate_on_gpu "${#CHECKPOINTS[@]}"
fi
