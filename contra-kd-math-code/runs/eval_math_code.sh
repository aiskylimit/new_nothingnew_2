#!/usr/bin/env bash
set -euo pipefail

# Usage: bash runs/eval_math_code.sh [--list] [BASE_PATH] [FAMILY] [METHOD] [VARIANT]
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST_ONLY=0
if [[ ${1:-} == --list ]]; then
    LIST_ONLY=1
    shift
fi

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

require_model() {
    local path=$1
    [[ -f "${path}/config.json" ]] || fail "Missing model config: ${path}/config.json"
    [[ -f "${path}/model.safetensors" || -f "${path}/model.safetensors.index.json" ||
       -f "${path}/pytorch_model.bin" || -f "${path}/pytorch_model.bin.index.json" ]] ||
        fail "Missing model weights in ${path}"
}

case "${FAMILY}" in all|gpt2|openllama2) ;; *) fail "Unknown family: ${FAMILY}" ;; esac
case "${METHOD}" in all|distillm|distillm2|csd|fdd) ;; *) fail "Unknown method: ${METHOD}" ;; esac
case "${VARIANT}" in all|baseline|contra) ;; *) fail "Unknown variant: ${VARIANT}" ;; esac
case "${EVAL_BACKEND}" in vllm|hf) ;; *) fail "Unknown evaluation backend: ${EVAL_BACKEND}" ;; esac

CHECKPOINTS=()
FAMILIES=()
while read -r family method variant checkpoint; do
    [[ ${FAMILY} == all || ${FAMILY} == "${family}" ]] || continue
    [[ ${METHOD} == all || ${METHOD} == "${method}" ]] || continue
    [[ ${VARIANT} == all || ${VARIANT} == "${variant}" ]] || continue
    if [[ ${family} == gpt2 && ${variant} == contra ]]; then
        checkpoint="${GPT2_CONTRA_DIR}/${checkpoint}"
    fi
    CHECKPOINTS+=("${checkpoint}")
    FAMILIES+=("${family}")
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
CHECKPOINTS

if [[ ${FAMILY} != gpt2 && -z ${OPENLLAMA2_MERGED_ROOT} ]]; then
    echo "OpenLLaMA2 base model: ${OPENLLAMA2_BASE_MODEL}"
    if [[ ${LIST_ONLY} == 0 ]]; then
        require_model "${OPENLLAMA2_BASE_MODEL}"
    fi
fi

MODEL_PATHS=()
for i in "${!CHECKPOINTS[@]}"; do
    family=${FAMILIES[$i]}
    checkpoint=${CHECKPOINTS[$i]}
    model_path="${BASE_PATH}/results/${family}/train/${checkpoint}"
    if [[ ${family} == openllama2 && -n ${OPENLLAMA2_MERGED_ROOT} ]]; then
        model_path="${OPENLLAMA2_MERGED_ROOT}/${checkpoint}"
    fi
    MODEL_PATHS+=("${model_path}")
    if [[ ${LIST_ONLY} == 1 ]]; then
        printf '%s\t%s\n' "${family}" "${model_path}"
    elif [[ ${family} == openllama2 && -z ${OPENLLAMA2_MERGED_ROOT} ]]; then
        [[ -f "${model_path}/adapter_config.json" ]] ||
            fail "Missing LoRA config: ${model_path}/adapter_config.json"
        [[ -f "${model_path}/adapter_model.safetensors" || -f "${model_path}/adapter_model.bin" ]] ||
            fail "Missing LoRA weights in ${model_path}"
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
command -v lm-eval >/dev/null || fail "Provision ../contra.txt and activate the contra environment."

export HF_HOME=${HF_HOME:-"${BASE_PATH}/.cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1
export HF_ALLOW_CODE_EVAL=1
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
    local model_path=${MODEL_PATHS[$i]}
    local model_args output_dir status
    model_args="pretrained=${model_path},dtype=float16"
    output_dir="${EVAL_OUTPUT_DIR}/${EVAL_BACKEND}/${family}/${CHECKPOINTS[$i]}"
    if [[ ${family} == openllama2 && -z ${OPENLLAMA2_MERGED_ROOT} ]]; then
        model_args="pretrained=${OPENLLAMA2_BASE_MODEL},dtype=float16"
        if [[ -f "${model_path}/tokenizer.json" || -f "${model_path}/tokenizer.model" ]]; then
            model_args+=",tokenizer=${model_path}"
        fi
        if [[ ${EVAL_BACKEND} == vllm ]]; then
            model_args+=",lora_local_path=${model_path},max_lora_rank=${MAX_LORA_RANK:-16}"
        else
            model_args+=",peft=${model_path}"
        fi
    elif [[ ${family} == openllama2 ]]; then
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

evaluate_on_gpu() {
    local gpu_index=$1
    local i status=0
    set +m
    export CUDA_VISIBLE_DEVICES="${GPU_IDS[$gpu_index]}"
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

for i in "${!GPU_IDS[@]}"; do
    if (( i >= ${#CHECKPOINTS[@]} )); then
        break
    fi
    evaluate_on_gpu "${i}" &
    PIDS+=("$!")
done

status=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || status=$?
    unset 'PIDS[i]'
done
trap - EXIT
exit "${status}"
