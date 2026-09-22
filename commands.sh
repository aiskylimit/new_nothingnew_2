#1 +60
#sft
#v1

ls -d /mnt/local/_models/aiskylimit_new_nothingnew_2/DeepSeek-R1-Distill-Qwen-1.5B \
      /mnt/local/_data/aiskylimit_new_nothingnew_2/{s1K-1.1,aime24,aime25,MATH-500,aimo-validation-amc} \
      /mnt/local/uvenvs/spectral_guided_learning{,_train}
cd SegmentSelectiveSFT && GPU=0 bash commands.sh
# CUDA_VISIBLE_DEVICES=1 bash project_commands.sh
# cd ./offline_olmo7b_b200
# bash project_commands.sh
# cd SpectralGuidedLearning && GPUS="0 1" bash project_commands_trans.sh
# kill -9 155157 155158

# cd ./offline_rlsd_sdpo_b200
# ls


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# SFT Long CoT -- DeepSeek-R1-Distill-Qwen-1.5B (s1K-1.1, full fine-tuning, eval thinking-off @4k):
# data -> train -> eval -> compare. Comment out once the run is done.
cd "$(dirname "${BASH_SOURCE[0]}")/SpectralGuidedLearning" \
  && GPUS=0 bash project_commands_r1-qwen-1.5b.sh


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


# cd ./multi-mode-distill
# bash ./project_commands.sh
# # # bash ./project_commands_opsd_ablation.sh
# # # bash ./project_commands_ablation.sh

