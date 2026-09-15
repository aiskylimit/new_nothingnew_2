#!/usr/bin/env bash
set -Eeuo pipefail

nvidia-smi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

cd ./sdxl_q3_offline_b200_2gpu
export GPU_IDS=2,3
export NUM_GPUS=2
export TARGET_GPU_FAMILY=B200
export TARGET_VRAM_PERCENT=95
export AUTOTUNE_STEPS=10

bash hessian/tune_b200_batch.sh
source runtime/b200-autotune.env

export PIPELINE_MODE=pilot
export PILOT_TRAIN_STEPS=10
export DATASET_PAIRS=4096
export OFFLINE_EVAL_LIMIT=2
export RUN_NAME="q3_dspo_sdxl_b200x2_tuned_mb${TRAIN_BATCH_SIZE}_pilot"
bash project_command.sh
