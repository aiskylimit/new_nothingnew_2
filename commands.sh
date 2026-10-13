#d
#models
--hf Qwen/Qwen2.5-3B-Instruct /mnt/local/@PROJECT@/grit/models/Qwen2.5-3B-Instruct
--hf Qwen/Qwen3Guard-Gen-4B /mnt/local/@PROJECT@/grit/models/Qwen3Guard-Gen-4B
#datasets
--hf-dataset fushinguyenex/GRIT-artifacts /mnt/local/@PROJECT@/grit/prepared
--hf-dataset cais/mmlu /mnt/local/@PROJECT@/grit/data/raw/mmlu
--hf-dataset m-a-p/SuperGPQA /mnt/local/@PROJECT@/grit/data/raw/SuperGPQA
--hf-dataset openai/gsm8k /mnt/local/@PROJECT@/grit/data/raw/gsm8k
--hf-dataset HuggingFaceH4/MATH-500 /mnt/local/@PROJECT@/grit/data/raw/MATH-500
--hf-dataset Hothan/OlympiadBench /mnt/local/@PROJECT@/grit/data/raw/OlympiadBench
--url https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test.jsonl /mnt/local/@PROJECT@/grit/data/raw/livecodebench
--url https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json /mnt/local/@PROJECT@/grit/data/raw/alpaca_eval
--hf-dataset Paul/XSTest /mnt/local/@PROJECT@/grit/data/raw/XSTest
--url https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-hard-1k.csv /mnt/local/@PROJECT@/grit/data/raw/or-bench
--url https://huggingface.co/datasets/bench-llm/or-bench/resolve/main/or-bench-toxic.csv /mnt/local/@PROJECT@/grit/data/raw/or-bench

#vlm
#v1

# cd SpectralGuidedLearning && bash project_commands_qwen25_7b_l100_t1_c2_const.sh
nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json-v2/ +a


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

# cd ./learn2follow
# bash ./project_commands.sh

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