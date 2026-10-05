#!/bin/bash
# Re-eval RLSD and SDPO on OLMo-3-7B-Think using checkpoints already on the B200 server.
# The two methods run in parallel on separate GPU pairs; steps within a method run sequentially.
# Outputs go to *_reeval directories so the original eval logs/results are not overwritten.
set -uo pipefail

ROOT=/mnt/local/aiskylimit_new_nothingnew_2/offline_olmo7b_b200
MODEL=/mnt/local/aiskylimit_new_nothingnew_2/tropic_baselines/models/Olmo-3-7B-Think
BENCHMARKS="aime25 aime26 hmmt25"
STEPS="25 50 75 100"

export TROPIC_TRAIN_DATA_PATH=/mnt/local/aiskylimit_new_nothingnew_2/tropic_baselines/data/train
export TROPIC_EVAL_DATA_DIR=/mnt/local/aiskylimit_new_nothingnew_2/tropic_baselines/data/eval
source /mnt/local/uvenvs/tropic/bin/activate
cd "$ROOT"

reeval_chain() {
  local script=$1 tag=$2 gpus=$3 port=$4 main=$5
  local outdir="$ROOT/results_${tag}_olmo7b_reeval"
  mkdir -p "$outdir"
  for step in $STEPS; do
    local ckpt="$ROOT/results_${tag}_olmo7b/${tag}_checkpoint_step${step}"
    local log="${outdir}/${tag}_reeval_step${step}.log"
    if [ ! -d "$ckpt" ]; then echo "skip (no checkpoint): $ckpt"; continue; fi
    if [ -f "$log" ] && grep -q "ALL DONE" "$log"; then echo "skip (already done): $log"; continue; fi
    env CUDA_VISIBLE_DEVICES="$main" python "scripts/${script}" \
      --model-path "$MODEL" --skip-train --checkpoint-path "$ckpt" \
      --output-dir "$outdir" \
      --eval-engine vllm --vllm-gpu-ids "$gpus" --vllm-base-port "$port" \
      --vllm-executable vllm --vllm-gpu-memory-utilization 0.9 \
      --vllm-max-model-len 41000 --eval-benchmarks $BENCHMARKS \
      2>&1 | tee "$log"
  done
}

reeval_chain run_rlsd_experiment_olmo7b.py rlsd "0,1" 8100 0 &
reeval_chain run_sdpo_experiment_olmo7b.py sdpo "2,3" 8200 2 &
wait
echo "ALL DONE - re-eval RLSD and SDPO finished"
