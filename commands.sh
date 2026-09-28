#d
#datasets
--hf-dataset siyanzhao/Openthoughts_math_30k_opsd /mnt/local/@PROJECT@/tropic_baselines/data/train
--hf-dataset yentinglin/aime_2025 /mnt/local/@PROJECT@/tropic_baselines/data/eval/aime25
--hf-dataset MathArena/aime_2026 /mnt/local/@PROJECT@/tropic_baselines/data/eval/aime26
--hf-dataset MathArena/hmmt_feb_2025 /mnt/local/@PROJECT@/tropic_baselines/data/eval/hmmt25
#models
--hf allenai/Olmo-3-7B-Think /mnt/local/@PROJECT@/tropic_baselines/models/Olmo-3-7B-Think

#vlm_distill_baseline
#v1


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN


# cd ./VLM_Distillation-main
# bash project_commands.sh
