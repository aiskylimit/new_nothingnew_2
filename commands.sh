#1 +90
#tropic
#v1

# OLMo-3-7B-Think training seeds 1-2 for TROPIC and OPSD (rebuttal), GPUs 2,3 - see offline_olmo_seeds_b200/.
# Model and data are the ones already downloaded for offline_olmo7b_b200 (same @PROJECT@ paths); if they are
# gone, first run a "#d" commit with the contents of offline_olmo_seeds_b200/download.txt.
# When finished, fetch the results (everything except the LoRA checkpoints) with:
# #2 -f-/mnt/local/aiskylimit_new_nothingnew_2/offline_olmo_seeds_b200/olmo_seeds_results.tgz

nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

cd ./offline_olmo_seeds_b200
bash ./project_commands.sh
