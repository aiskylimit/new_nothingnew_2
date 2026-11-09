#d
#datasets
--hf-dataset Elriggs/openwebtext-100k /mnt/local/@PROJECT@/contra-kd/data/raw/openwebtext-100k
--hf-dataset MiniLLM/dolly /mnt/local/@PROJECT@/contra-kd/data/raw/dolly
--hf-dataset MiniLLM/self-inst /mnt/local/@PROJECT@/contra-kd/data/raw/self-inst
--hf-dataset MiniLLM/Vicuna /mnt/local/@PROJECT@/contra-kd/data/raw/vicuna
--hf-dataset MiniLLM/sinst /mnt/local/@PROJECT@/contra-kd/data/raw/sinst
#models
--hf bachthetrollface/gpt2-120M-fdd-dolly /mnt/local/@PROJECT@/contra-kd/results/gpt2/train/fdd/base
--hf bachthetrollface/gpt2-120M-contra-fdd-dolly /mnt/local/@PROJECT@/contra-kd/results/gpt2/train/contra/fdd/base
--hf bachthetrollface/gpt2-120M-distillm-dolly /mnt/local/@PROJECT@/contra-kd/results/gpt2/train/distillm/base
--hf bachthetrollface/gpt2-120M-contra-distillm-dolly /mnt/local/@PROJECT@/contra-kd/results/gpt2/train/contra/distillm/base

#contra1
#v1


nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json-v2/ +a


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py



# ls OPSD/results/raw/olmo3-7b-think/opsd/
# ls OPSD/res_tar
# mkdir -p ./res_tar
# cd OPSD
# tar -czf - ./results/raw/ | split -b 24M - ./res_tar/opsd_2.tar.gz.part-

# cd ./learn2follow
# bash ./project_commands.sh

cd ./contra-kd
bash ./project_commands.sh

# cd ./VLM_Distillation-main
# bash project_commands_collect_eval_summary.sh
# bash project_commands.sh
# pwd && ls 

# cd ./opsd
# bash ./project_commands.sh

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./grit
# bash ./project_commands.sh