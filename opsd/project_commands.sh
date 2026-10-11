#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /mnt/local/uvenvs/opsd/bin/activate

# Must match the destinations in download.txt.
PROJECT_NAME="aiskylimit_new_nothingnew_2"
BASE_DIR="/mnt/local/${PROJECT_NAME}/OPSD"
export MODEL_ROOT="${BASE_DIR}/models"
export RAW_DATA_ROOT="${BASE_DIR}/data/raw"
export PREPARED_DATA_ROOT="${BASE_DIR}/data/processed"
RUN_TAG="sft_qwen3_8b_4gpu_$(date +%Y%m%d_%H%M%S)_$$"
export OUTPUT_ROOT="${BASE_DIR}/outputs/${RUN_TAG}"
export RESULTS_ROOT="${BASE_DIR}/results/seed42/${RUN_TAG}"
export HF_HOME="${BASE_DIR}/.cache/huggingface"

# Four-GPU SFT training: 4 examples x 4 GPUs x 2 accumulation = batch 32.
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_PROCESSES=4
export PER_DEVICE_BATCH_8B=4
export GRADIENT_ACCUMULATION_8B=2
export EVAL_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.6
export GPU_MEMORY_UTILIZATION=0.9
export MAIN_PROCESS_PORT=auto

# Prepare only the datasets this run needs; keep already prepared data intact.
if [[ ! -d "${PREPARED_DATA_ROOT}/train" ]]; then
    python "${PROJECT_ROOT}/data/prepare_data.py" \
        --raw_root "${RAW_DATA_ROOT}" \
        --output_root "${PREPARED_DATA_ROOT}" \
        --only_train
fi
if [[ ! -d "${PREPARED_DATA_ROOT}/eval/aime26" ]]; then
    python "${PROJECT_ROOT}/data/prepare_data.py" \
        --raw_root "${RAW_DATA_ROOT}" \
        --output_root "${PREPARED_DATA_ROOT}" \
        --only_eval aime26
fi

# Fresh SFT training followed by AIME26 evaluation at seed 42.
export SEED=42
export EVAL_SEED=42
export EVAL_VAL_N=12
export OVERWRITE_EVAL=0

echo "Training output: ${OUTPUT_ROOT}"
echo "Evaluation results: ${RESULTS_ROOT}"
bash "${PROJECT_ROOT}/scripts/run_training.sh" sft 8b
EVAL_DATASETS="aime26" bash "${PROJECT_ROOT}/eval/run_eval_matrix.sh" 8b sft
