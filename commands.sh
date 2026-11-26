#d
#datasets
--hf-dataset simplescaling/s1K-1.1 /mnt/local/_data/@PROJECT@/s1K-1.1
--hf-dataset Maxwell-Jia/AIME_2024 /mnt/local/_data/@PROJECT@/AIME_2024
--hf-dataset yentinglin/aime_2025 /mnt/local/_data/@PROJECT@/aime_2025
--hf-dataset HuggingFaceH4/MATH-500 /mnt/local/_data/@PROJECT@/MATH-500
--hf-dataset AI-MO/aimo-validation-amc /mnt/local/_data/@PROJECT@/aimo-validation-amc
#models
--hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B /mnt/local/_models/@PROJECT@/DeepSeek-R1-Distill-Qwen-1.5B
--hf Qwen/Qwen2.5-7B-Instruct /mnt/local/_models/@PROJECT@/Qwen2.5-7B-Instruct
--hf Qwen/Qwen3-8B /mnt/local/_models/@PROJECT@/Qwen3-8B

#alerd
#v1


nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json-v2/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/contra-kd/results/gpt2/eval_main/results/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/contra-kd-2/results/gpt2/eval_main/results/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/contra-kd-math-code/results/eval_math_code_export/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/contra-kd-datdvt/logs/gpt2_direct_mlp_ablation/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/contra-kd-datdvt/logs/gpt2_loss_ramp_ablation/ +a
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/cred/MMEB-evaloutputs-json-v5/ +a

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./contra-kd-datdvt
# bash ./project_commands.sh nonuniform

# cd ./contra-kd-2
# bash ./project_commands.sh fdd-seed-study

# cd ./contra-kd-math-code
# bash ./project_commands.sh .

# cd ./VLM_Embed_Gvendi
# bash ./project_commands.sh

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

