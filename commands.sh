#2 -0-10
#ssft_8b
#v1

# tree SegmentSelectiveSFT
# cd P-ALIGN
# nvidia-smi
# cd SpectralGuidedLearning && GPUS=0 bash project_commands2.sh
# cd SegmentSelectiveSFT && GPU=1 bash commands.sh
nvidia-smi
# cd SegmentSelectiveSFT && GPU=0 bash commands.sh
# CUDA_VISIBLE_DEVICES=1 bash project_commands.sh
# cd ./offline_olmo7b_b200
# bash project_commands.sh
# nvidia-smi
# cd SpectralGuidedLearning && GPUS=0 bash scripts/eval/reeval_all.sh

# kill -9 155157 155158

# cd ./offline_rlsd_sdpo_b200
# ls


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# The platform GPU guard keeps otherwise-idle devices busy with a synthetic
# burner. Stop only that known helper; never terminate arbitrary CUDA jobs.
# guard_pids="$(pgrep -f '[/]tmp/llm_pretrain_burn.py' || true)"
# if [[ -n "$guard_pids" ]]; then
#   echo "Stopping GPU guard burners: $guard_pids"
#   ps -fp $guard_pids || true
#   kill -TERM $guard_pids 2>/dev/null || true
#   sleep 5
# fi
# nvidia-smi

# cd ./sdxl_q3_offline_b200_2gpu
# export GPU_IDS=1,2,6,7
# export NUM_GPUS=4
# export TARGET_GPU_FAMILY=B200
# export TARGET_VRAM_PERCENT=94
# export AUTOTUNE_STEPS=10
# export B200_MAX_PREEXISTING_MEMORY_MIB=8192
# export B200_BATCH_CANDIDATES="24 32 40 48 56 64"

# bash hessian/tune_b200_batch.sh
# source runtime/b200-autotune.env

# export PIPELINE_MODE=pilot
# export PILOT_TRAIN_STEPS=10
# export DATASET_PAIRS=4096
# export OFFLINE_EVAL_LIMIT=2
# export RUN_NAME="q3_dspo_sdxl_b200x4_tuned_mb${TRAIN_BATCH_SIZE}_pilot"
# bash project_command.sh

# cd ./offline_rlsd_sdpo_b200
# bash project_commands.sh

# cd ./opsd
# bash project_commands.sh

# cd ./VLM_Distillation-main
# bash project_commands.sh
