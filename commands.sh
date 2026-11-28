#1 +30
#alerd-log
#v4


# ---- tiến độ + kết quả run 4 (7B) và run 7 (8B), chỉ đọc
cd SpectralGuidedLearning
LINES=5 bash phase3_status.sh
for m in qwen25-7b qwen3-8b; do
  n="iwc-gain-l05-lora-${m}-s1k"
  echo; echo "======== ${m}: lỗi trong lane log mới nhất"
  tr '\r' '\n' < "$(ls -t rebuttal/logs/phase3-gpu*-"${m}"-*.log | head -1)" |
    grep -E "failed with exit code|Traceback|Error|FAILED" | tail -n 10 | cut -c1-200
  echo "======== ${m}: kết quả eval"
  cat "rebuttal/results/${n}/summary.json" 2>/dev/null || echo "(chưa có)"
  echo
done
cd ..

# cd ./offline_olmo_seeds_b200
# bash ./project_commands.sh

# cd ./offline_olmo_seeds_b200_rev
# bash ./project_commands.sh

nvidia-smi
#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/talas_vlm_embed/MMEB-evaloutputs-json-v2/ +a

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./OpenED
# bash ./project_commands_23.sh

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
