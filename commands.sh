#2 +a
#contra3
#v1

#d
#datasets
# --hf-dataset TIGER-Lab/MMEB-train /mnt/local/@PROJECT@/cred/vlm2vec_train/MMEB-train
# --hf-dataset VoCuc/vlm-teacher-embedding /mnt/local/@PROJECT@/cred/datasets
# --hf-dataset TIGER-Lab/MMEB-eval /mnt/local/@PROJECT@/cred/vlm2vec_eval/MMEB-eval
# --url https://huggingface.co/VoCuc/en_core_web_sm/resolve/main/en_core_web_sm.zip /mnt/local/@PROJECT@/cred/en_core_web_sm.zip
#models
# --hf raghavlite/B3_Qwen2_2B /mnt/local/@PROJECT@/cred/models/B3_Qwen2_2B
# --hf apple/FastVLM-0.5B /mnt/local/@PROJECT@/cred/models/FastVLM-0.5B
# --hf apple/FastVLM-1.5B /mnt/local/@PROJECT@/cred/models/FastVLM-1.5B
# --hf llava-hf/llava-onevision-qwen2-0.5b-ov-hf /mnt/local/@PROJECT@/cred/models/llava-onevision-qwen2-0.5b-ov-hf



nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json-v2/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/contra-kd/logs/gpt2_t0/,/mnt/local/aiskylimit_new_nothingnew_2/contra-kd/results/gpt2/eval_main/ +a


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./contra-kd
# bash ./project_commands.sh
# kill 7474

# cd ./contra-kd-2
# bash ./project_commands.sh

# ls OPSD/results/raw/olmo3-7b-think/opsd/
# ls OPSD/res_tar
# mkdir -p ./res_tar
# cd OPSD
# tar -czf - ./results/raw/ | split -b 24M - ./res_tar/opsd_2.tar.gz.part-

# cd ./learn2follow
# bash ./project_commands.sh

# cd ./contra-kd-datdvt
# bash ./project_commands.sh

# cd ./cred
# bash ./project_commands.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh