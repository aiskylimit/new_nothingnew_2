#d
#datasets
--url https://huggingface.co/datasets/datht/processed-cl-ace/resolve/main/ace_all.tar.gz /mnt/local/@PROJECT@/OpenED/ace_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-maven/resolve/main/maven_all.tar.gz /mnt/local/@PROJECT@/OpenED/maven_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-rams/resolve/main/rams_all.tar.gz /mnt/local/@PROJECT@/OpenED/rams_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-geneva/resolve/main/geneva_all.tar.gz /mnt/local/@PROJECT@/OpenED/geneva_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-tacred/resolve/main/tacred_all.tar.gz /mnt/local/@PROJECT@/OpenED/tacred_all.tar.gz
--url https://huggingface.co/datasets/datht/processed-cl-fewrel/resolve/main/fewrel_all.tar.gz /mnt/local/@PROJECT@/OpenED/fewrel_all.tar.gz
#models
--hf Qwen/Qwen3-0.6B /mnt/local/@PROJECT@/OpenED/models/Qwen3-0.6B
--hf meta-llama/Llama-3.2-1B-Instruct /mnt/local/@PROJECT@/OpenED/models/Llama-3.2-1B-Instruct
--hf google/gemma-3-1b-it /mnt/local/@PROJECT@/OpenED/models/gemma-3-1b-it

#gpu
#v1

# kill -9 213897
# cd SpectralGuidedLearning && bash project_commands_b200_gain.sh 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py
nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# ls OPSD/results/raw/olmo3-7b-think/opsd/
# ls OPSD/res_tar
# mkdir -p ./res_tar
# cd OPSD
# tar -czf - ./results/raw/ | split -b 24M - ./res_tar/opsd_2.tar.gz.part-

# cd ./VLM_Distillation-main
# bash project_commands_collect_eval_summary.sh
# bash project_commands.sh

# cd ./offline_olmo7b_b200
# bash ./project_commands.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./talas_vlm_embed
# bash ./project_commands.sh