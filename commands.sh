#d
#datasets
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/coco/train2017.zip /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/train_data/coco
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/gqa/images.zip /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/train_data/gqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/ocr_vqa/ocr_vqa_images.zip /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/train_data/ocr_vqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/ocr_vqa/dataset.json /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/train_data/ocr_vqa
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/vg/images.zip /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/train_data/vg
--url https://huggingface.co/datasets/DVLe/llava_dataset/resolve/main/vg/images2.zip /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/train_data/vg
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/GQA_TestDev_Balanced.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MME.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/RealWorldQA.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/ScienceQA_TEST.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/AI2D_TEST_NO_MASK.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MMMU_DEV_VAL.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/MMStar.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/ChartQA_TEST.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/DocVQA_VAL.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/TextVQA_VAL.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
--url https://huggingface.co/datasets/DVLe/Eval_VLM/resolve/main/OCRBench.tsv /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/eval_data/LMUData
#models
--hf Qwen/Qwen2.5-VL-3B-Instruct /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-3B-Instruct
--hf Qwen/Qwen3-VL-8B-Instruct /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/Qwen/Qwen3-VL-8B-Instruct
--hf KamilaMila/FastVLM-0.5B /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/KamilaMila/FastVLM-0.5B
--hf Qwen/Qwen2-VL-2B-Instruct /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/Qwen/Qwen2-VL-2B-Instruct
--hf Qwen/Qwen2.5-VL-7B-Instruct /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/Qwen/Qwen2.5-VL-7B-Instruct
--hf Qwen/Qwen3-VL-4B-Instruct /mnt/local/aiskylimit_new_nothingnew_2/VLM_Distillation-main/models/Qwen/Qwen3-VL-4B-Instruct

#vlm-distill
#v1

# cd P-ALIGN
nvidia-smi
cd ./OPSD
tar -czvf opsd_4b_results.tar.gz results/raw/qwen3-4b/opsd
# CUDA_VISIBLE_DEVICES=1 bash project_commands.sh

# nvidia-smi
# SDXL_ENV=/mnt/local/uvenvs/sdxl-q3-offline-b200-2gpu
# if [[ ! -x "$SDXL_ENV/bin/python" ]]; then
#   echo "SDXL_ENV_MISSING=$SDXL_ENV"
#   exit 1
# fi
# echo "SDXL_ENV_READY=$SDXL_ENV"
# "$SDXL_ENV/bin/python" --version
# nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
# nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
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