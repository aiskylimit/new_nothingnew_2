#1 +60
#tropic_rev
#v1

# SECOND RUNNER for the OLMo-3-7B-Think seeds 1-2 evals, on GPUs 4,5 - see offline_olmo_seeds_b200_rev/.
# Runs NEXT TO the job already going on GPUs 2,3 (offline_olmo_seeds_b200/, do not stop it): it evaluates the
# remaining checkpoints of that job starting from the END of its list (OPSD seed 2 step 100 first), writes into
# the same result folders, and never takes a checkpoint the first job is evaluating or has finished.
# Nothing is trained or downloaded here. When both are finished, fetch the results with:
# #2 -f-/mnt/local/aiskylimit_new_nothingnew_2/offline_olmo_seeds_b200/olmo_seeds_results.tgz

nvidia-smi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

cd ./offline_olmo_seeds_b200_rev
bash ./project_commands.sh
