#1 +60
#tropic_rev
#v1

# cd SpectralGuidedLearning
# /mnt/local/uvenvs/spectral_guided_learning/bin/python scripts/report_run1.py
# for a in iwc-gain-t2-l05 iwc-gain-t1-l03 iwc-gain-t1-l07 iwc-gain-t2-l10; do
#   tag=$a-r1-qwen-1.5b-s1k
#   if [ -f rebuttal/tune/results/$tag/summary.json ]; then
#     echo; echo "== $tag: xong"
#   else
#     echo; echo "== $tag: CHƯA có điểm, đuôi log:"
#     log=$(ls -t rebuttal/tune/logs/$tag-*.log 2>/dev/null | head -1)
#     if [ -n "$log" ]; then echo "$log"; tail -n 15 "$log"; else echo "(chưa có log)"; fi
#   fi
# done

# cd ./offline_olmo_seeds_b200
# bash ./project_commands.sh

cd ./offline_olmo_seeds_b200_rev
bash ./project_commands.sh

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
