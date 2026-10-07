#2 -0-10
#iwc
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

# cd ./OpenED
# for name in ace_all maven_all rams_all geneva_all tacred_all fewrel_all; do
#     dir="${name}.tar.gz"
#     tmp="/tmp/${name}.tar.gz"

#     mv "$dir/$dir" "$tmp" &&
#     rm -rf "$dir" &&
#     mv "$tmp" "$dir"

#     echo "Done: $dir"
# done
# bash ./project_commands_7.sh