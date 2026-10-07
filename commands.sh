#1 +10
#setup
#v1



#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json/ +a


# cd SpectralGuidedLearning && cat results_b200/summary-iwc-gain-nocap.md
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

cd ./talas_vlm_embed
mv models/Qwen/Qwen/Qwen2-VL-7B-Instruct models/Qwen/
bash ./project_commands.sh

# cd ./OpenED
# bash gather_logs.sh
# bash ./project_commands_7.sh