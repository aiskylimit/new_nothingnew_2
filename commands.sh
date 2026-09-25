#i reasoning-velocity-distill.txt
#i
#v1


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# nvidia-smi

# cd SegmentSelectiveSFT && bash commands.sh
# cd P-ALIGN && CUDA_VISIBLE_DEVICES=1 bash project_commands_r1_1.5b.sh
# cd SegmentSelectiveSFT && GPU=0 bash commands.sh
# CUDA_VISIBLE_DEVICES=1 bash project_commands.sh
# cd ./offline_olmo7b_b200
# bash project_commands.sh
# cd P-ALIGN && GPUS=1 bash  project_commands_r1_1.5b.sh
# kill -9 155157 155158
# cd ./SpectralGuidedLearning && GPUS=0 bash project_commands_iwc_r1-qwen-1.5b.sh 
# cd ./offline_rlsd_sdpo_b200
# ls


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN


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
# bash ./run_gemma.sh
# bash ./run_gemma_ablations.sh
# tree /mnt/local/aiskylimit_new_nothingnew_2/multi-mode-distill/results/gemma-2-2b-it-distill/geo1_cka0_menger0.0/e2-bs8-lr0.0001-G4-N4-NN1-kd0.5-lora-16-128-0.05/evaluation_gemma
# bash ./project_commands.sh
# # # bash ./project_commands_opsd_ablation.sh
# # # bash ./project_commands_ablation.sh
# cat /mnt/local/aiskylimit_new_nothingnew_2/multi-mode-distill/results/gemma-2-2b-it-distill/geo1_cka0_menger0.0/e2-bs8-lr0.0001-G4-N4-NN1-kd0.5-lora-16-128-0.05/evaluation_gemma/code/__mnt__local__aiskylimit_new_nothing__reasoning_velocity_distill__models__google_gemma-2-2b-it/samples_mbpp_2026-09-24T07-21-52.951985.jsonl
# cat /mnt/local/aiskylimit_new_nothingnew_2/multi-mode-distill/results/gemma-2-2b-it-distill/geo1_cka0_menger0.0/e2-bs8-lr0.0001-G4-N4-NN1-kd0.5-lora-16-128-0.05/evaluation_gemma/code/__mnt__local__aiskylimit_new_nothing__reasoning_velocity_distill__models__google_gemma-2-2b-it/samples_mbpp_2026-09-24T14-54-48.288245.jsonl
