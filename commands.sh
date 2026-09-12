#i spectral_guided_learning.txt
#set_up_spectral_guided_learning
#v2

# source /mnt/local/uvenvs/spectral-guided-learning/bin/activate
# cd ./SpectralGuidedLearning && bash ./project_commands.sh


#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v3/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/_run_log_/_run-2026-09-03_17-01-16-VLM-Distillation.log
#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/eval/ +a

# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=6,7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 3
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# source /mnt/local/uvenvs/talas-vlm-embed/bin/activate
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./talas_vlm_embed/multi_gpu_v2.py

# cd ./talas_vlm_embed
# bash ./project_commands.sh


# cd ./VLM_Distillation-main
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./project_commands.sh


# cd ./sdxl_q3_offline_b200_2gpu
# CUDA_VISIBLE_DEVICES=2,3 bash ./project_command.sh


# cd ./SegmentSelectiveSFT
# CUDA_VISIBLE_DEVICES=0,1 bash commands.sh
