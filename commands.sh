#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/cred/MMEB-evaloutputs-json-v5/ +a
#contra-math-code
#v2

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
# bash ./project_commands.sh direct-mlp-distillm-eval

# cd ./contra-kd-2
# bash ./project_commands.sh

# cd ./contra-kd-math-code
# bash ./project_commands.sh .

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

# mkdir -p /mnt/local/aiskylimit_new_nothingnew_2/rebuttal_cored/rebuttal
# mv /mnt/local/aiskylimit_new_nothingnew_2/rebuttal/assets \
#    /mnt/local/aiskylimit_new_nothingnew_2/rebuttal_cored/rebuttal/

# cd rebuttal_cored
# bash rebuttal/project_commands.sh prepare && \
# GPUS=0,1,2,3 bash rebuttal/project_commands.sh preflight && \
# GPUS=0,1,2,3 bash rebuttal/project_commands.sh smoke --settings C && \
# GPUS=0,1,2,3 bash rebuttal/project_commands.sh train --settings A C && \
# GPUS=0,1,2,3 bash rebuttal/project_commands.sh eval --settings A C && \
# bash rebuttal/project_commands.sh report --allow-partial
