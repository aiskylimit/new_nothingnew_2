#d
#datasets
--hf-dataset siyanzhao/Openthoughts_math_30k_opsd /mnt/local/@PROJECT@/OPSD/data/raw/train
--hf-dataset HuggingFaceH4/aime_2024 /mnt/local/@PROJECT@/OPSD/data/raw/eval/aime24
--hf-dataset yentinglin/aime_2025 /mnt/local/@PROJECT@/OPSD/data/raw/eval/aime25
--hf-dataset MathArena/aime_2026 /mnt/local/@PROJECT@/OPSD/data/raw/eval/aime26
--hf-dataset MathArena/hmmt_feb_2025 /mnt/local/@PROJECT@/OPSD/data/raw/eval/hmmt25
#models
--hf Qwen/Qwen3-4B /mnt/local/@PROJECT@/OPSD/models/Qwen3-4B
--hf Qwen/Qwen3-8B /mnt/local/@PROJECT@/OPSD/models/Qwen3-8B

#opsd
#v2

# cd SpectralGuidedLearning && bash project_commands_qwen25_7b_l100_t1_c2_const.sh
nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json/ +a


# cd SpectralGuidedLearning && cat results_b200/summary-iwc-gain-nocap.md
# cd SpectralGuidedLearning && bash project_commands_report_e9e10_qwen25_tune.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py


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

# cd ./OpenED
# bash gather_logs.sh all3
# tar -czf collected_logs.tar.gz collected_logs/all3/
# ls -lh collected_logs.tar.gz
# GPUS="0 1" bash ./project_commands_8.sh