#d
#datasets
--hf-dataset simplescaling/s1K-1.1 /mnt/local/_data/@PROJECT@/s1K-1.1
--hf-dataset GAIR/LIMO /mnt/local/_data/@PROJECT@/LIMO
--hf-dataset math-ai/aime24 /mnt/local/_data/@PROJECT@/aime24
--hf-dataset math-ai/aime25 /mnt/local/_data/@PROJECT@/aime25
--hf-dataset HuggingFaceH4/MATH-500 /mnt/local/_data/@PROJECT@/MATH-500
--hf-dataset AI-MO/aimo-validation-amc /mnt/local/_data/@PROJECT@/aimo-validation-amc
--hf-dataset liuhuohuo2/pick-a-pic-v2 /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/data/pickapic_v2_full
--hf-dataset nateraw/parti-prompts /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/eval_sources/parti-prompts
--url https://huggingface.co/datasets/ymhao/HPDv2/resolve/main/test.json /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/eval_sources/HPDv2/
#models
--hf deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B /mnt/local/_models/@PROJECT@/DeepSeek-R1-Distill-Qwen-1.5B
--hf deepseek-ai/DeepSeek-R1-Distill-Qwen-7B /mnt/local/_models/@PROJECT@/DeepSeek-R1-Distill-Qwen-7B
--hf Qwen/Qwen2.5-7B-Instruct /mnt/local/_models/@PROJECT@/Qwen2.5-7B-Instruct
--hf Qwen/Qwen3-8B /mnt/local/_models/@PROJECT@/Qwen3-8B
--hf stabilityai/stable-diffusion-xl-base-1.0 /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/models/stable-diffusion-xl-base-1.0
--hf madebyollin/sdxl-vae-fp16-fix /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/models/sdxl-vae-fp16-fix
--hf laion/CLIP-ViT-H-14-laion2B-s32B-b79K /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/CLIP-ViT-H-14-laion2B-s32B-b79K
--hf yuvalkirstain/PickScore_v1 /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/PickScore_v1
--hf google-bert/bert-base-uncased /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/bert-base-uncased
--url https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/HPSv2/
--url https://raw.githubusercontent.com/tgxs002/HPSv2/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/HPSv2/
--url https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/open_clip/
--url https://huggingface.co/trl-lib/ddpo-aesthetic-predictor/resolve/main/aesthetic-model.pth /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/ddpo-aesthetic-predictor/
--url https://huggingface.co/THUDM/ImageReward/resolve/main/ImageReward.pt /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/ImageReward/
--url https://huggingface.co/THUDM/ImageReward/resolve/main/med_config.json /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets/reward_models/ImageReward/
# #spectral_guided_learning
# #v1
# cd ./SpectralGuidedLearning
# bash ./project_commands.sh
#download_sdxl_q3_offline_assets
#v1

# The 10 spectral_guided_learning items above are kept intact and first in the
# block; the 14 sdxl_q3_offline_assets items are appended, so one download
# session serves both projects instead of either trigger replacing the other.
# Nothing else in this file changed except the SDXL notes and the df/ls below.
#
# Why the SDXL assets are being refetched: the df17a26 run cleared the uv-env
# gate installed by f2885fd and then died at
#   [FAILED] dataset directory missing:
#   /mnt/local/aiskylimit_new_nothingnew_2/sdxl_q3_offline_b200_2gpu/offline_assets/data/pickapic_v2_full/data
# env.sh had logged asset_root=$PROJECT_ROOT/offline_assets, its fallback branch,
# so neither the colocated root nor the legacy aiskylimit_new_nothing root has
# the data. /mnt/local is node-local, so the replacement node lost the offline
# assets together with the uv env.
#
# Destinations are /mnt/local/_data/@PROJECT@/sdxl_q3_offline_assets, the same
# platform dataset area the spectral items use, because it sits outside the git
# clone and survives a re-clone. env.sh now resolves that root first. Verified
# against the Hessian manifest for the same dataset: 645 train shards, 299.63
# GiB, 851,293 binary pairs. The SDXL share is ~410 GiB (311.85 GiB data/ tree,
# 71.63 GiB SDXL repo, ~26 GiB reward models).
#
# No #1 directive here, mirroring 5a050a1, which is the shape that produced
# "download: 9 item(s) OK". Training stays commented until _RUN_STATUS_.log
# reports this download complete; a follow-up commit then restores the #1 +10
# launch on GPUs 2,3.
echo "=== disk headroom before the ~410 GiB SDXL refetch ==="
df -h /mnt/local 2>&1
ls -la /mnt/local/_data 2>&1 | head
ls -la /mnt/local/_data/aiskylimit_new_nothingnew_2 2>&1 | head -20

# SDXL Q3 DSPO full851k on physical GPU 2,3.
# env.sh maps CUDA_VISIBLE_DEVICES -> GPU_IDS (2,3); NUM_GPUS=2, effective batch 64.
# The uv env /mnt/local/uvenvs/sdxl-q3-offline-b200-2gpu was installed by f2885fd.
# SpectralGuidedLearning stays commented: it runs in the foreground and cd's away,
# which would both block this job and break the relative cd below.
# cd ./sdxl_q3_offline_b200_2gpu
# CUDA_VISIBLE_DEVICES=2,3 \
# PIPELINE_MODE=full851k \
# RUN_NAME=q3_dspo_sdxl_full851k_eb64_2gpu_gpu23_restart_20260912 \
# bash ./project_command.sh

# cd ./SpectralGuidedLearning && bash ./project_commands.sh

#2 -f-/mnt/local/aiskylimit_new_nothingnew_2/sdxl_q3_offline_b200_2gpu/runtime/logs/
#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v3/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/_run_log_/_run-2026-09-03_17-01-16-VLM-Distillation.log
#2 -f-/mnt/local/aiskylimit_new_nothing/VLM_Distillation-main/outputs/eval/ +a

# nvidia-smi
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
# CUDA_VISIBLE_DEVICES=6,7 python3 /tmp/llm_pretrain_burn.py &

# kill -9 $(nvidia-smi -i 6,7 --query-compute-apps=pid --format=csv,noheader)
# sleep 3
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 /tmp/llm_pretrain_burn.py &
nvidia-smi


export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# source /mnt/local/uvenvs/talas-vlm-embed/bin/activate
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 ./talas_vlm_embed/multi_gpu_v2.py

# cd ./talas_vlm_embed
# bash ./project_commands.sh


# cd ./VLM_Distillation-main
# CUDA_VISIBLE_DEVICES=4,5,6,7 bash ./project_commands.sh


# cd ./sdxl_q3_offline_b200_2gpu
# CUDA_VISIBLE_DEVICES=2,3 bash ./project_command.sh


# cd ./SegmentSelectiveSFT
# CUDA_VISIBLE_DEVICES=0,1 bash commands.sh
