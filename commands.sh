#d
#datasets
--url https://huggingface.co/datasets/VoCuc/vlm-teacher-embedding/resolve/main/B3_Qwen2_2B_grounding.tar.gz /mnt/local/@PROJECT@/talas_vlm_embed/datasets
--url https://huggingface.co/datasets/VoCuc/vlm-teacher-embedding/resolve/main/B3_Qwen2_7B_cls.tar.gz /mnt/local/@PROJECT@/talas_vlm_embed/datasets
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/RefCOCO/test-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_eval/MMEB-eval/RefCOCO
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/RefCOCO-Matching/test-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_eval/MMEB-eval/RefCOCO-Matching
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/Visual7W-Pointing/test-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_eval/MMEB-eval/Visual7W-Pointing
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/MSCOCO/test-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_eval/MMEB-eval/MSCOCO
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/MSCOCO/diverse_instruction-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_train/MMEB-train/MSCOCO
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/MSCOCO/original-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_train/MMEB-train/MSCOCO
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/MSCOCO/train-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_train/MMEB-train/MSCOCO
--url https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/MSCOCO/test-00000-of-00001.parquet /mnt/local/@PROJECT@/talas_vlm_embed/vlm2vec_eval/MMEB-eval/MSCOCO
#models
--hf raghavlite/B3_Qwen2_7B /mnt/local/@PROJECT@/talas_vlm_embed/models/B3_Qwen2_7B
--hf Qwen/Qwen2-VL-7B-Instruct /mnt/local/@PROJECT@/talas_vlm_embed/models/Qwen/Qwen/Qwen2-VL-7B-Instruct

#log
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
JSON_FILTER_DESTINATION="${JSON_FILTER_DESTINATION:-./MMEB-evaloutputs-json}"
python json_filter.py ./MMEB-eval_outputs "${JSON_FILTER_DESTINATION}" --overwrite
# bash ./project_commands.sh

# cd ./OpenED
# bash gather_logs.sh
# bash ./project_commands_7.sh