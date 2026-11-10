#d
#datasets
--hf-dataset simplescaling/s1K-1.1 /mnt/local/@PROJECT@/rebuttal/assets/datasets/s1K-1.1
--hf-dataset Maxwell-Jia/AIME_2024 /mnt/local/@PROJECT@/rebuttal/assets/benchmarks/AIME_2024
--hf-dataset yentinglin/aime_2025 /mnt/local/@PROJECT@/rebuttal/assets/benchmarks/aime_2025
--hf-dataset AI-MO/aimo-validation-amc /mnt/local/@PROJECT@/rebuttal/assets/benchmarks/aimo-validation-amc
--hf-dataset HuggingFaceH4/MATH-500 /mnt/local/@PROJECT@/rebuttal/assets/benchmarks/MATH-500
--url https://huggingface.co/datasets/allenai/ai2_arc/resolve/210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/test-00000-of-00001.parquet /mnt/local/@PROJECT@/rebuttal/assets/benchmarks/ARC-Challenge/
--url https://raw.githubusercontent.com/csitfun/LogiQA2.0/31381f5634d201c4ed6f981b87de3c7d9f496002/logiqa/DATA/LOGIQA/test.txt /mnt/local/@PROJECT@/rebuttal/assets/benchmarks/LogiQA2.0/
#models
--hf Qwen/Qwen2.5-7B-Instruct /mnt/local/@PROJECT@/rebuttal/assets/models/Qwen2.5-7B-Instruct
--hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B /mnt/local/@PROJECT@/rebuttal/assets/models/DeepSeek-R1-Distill-Qwen-1.5B

#contra-direct-mlp
#v1


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

# cd ./contra-kd-datdvt
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