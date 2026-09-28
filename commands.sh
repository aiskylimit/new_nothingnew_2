#i vlm-distill-eval.txt
#vlm
#v1


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

nvidia-smi

# cd SegmentSelectiveSFT && bash commands.sh
# cd P-ALIGN && CUDA_VISIBLE_DEVICES=1 bash project_commands_r1_1.5b.sh
# cd SegmentSelectiveSFT && GPU=0 bash commands.sh
# CUDA_VISIBLE_DEVICES=1 bash project_commands.sh
# cd ./offline_olmo7b_b200
# bash project_commands.sh
# cd P-ALIGN && GPUS=1 bash  project_commands_r1_1.5b.sh
# kill -9 155157 155158
# cd ./SpectralGuidedLearning && GPUS=0 bash project_commands_iwc_r1-qwen-1.5b.sh 
# cd ./offline_rlsd_sdpo_b200
# ls


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN


# cd ./offline_rlsd_sdpo_b200
# bash project_commands.sh

# cd ./opsd
# bash project_commands.sh

# cd ./VLM_Distillation-main
# bash project_commands.sh


# cd ./multi-mode-distill
# bash ./project_commands.sh


# cd ./offline_olmo7b_b200
# bash ./project_commands.sh