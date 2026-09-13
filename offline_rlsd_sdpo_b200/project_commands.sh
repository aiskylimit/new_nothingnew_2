#!/usr/bin/env bash
# Full pipeline: RLSD + SDPO baselines on Qwen3-4B and Qwen3-8B, evaluated on
# AIME25/AIME26/HMMT25. This is the ONLY file meant to be run directly - it
# assumes download.txt has already been processed (models/datasets staged
# locally) and tropic.txt's env is already installed and active.
#
# Run this under something that survives a disconnect, e.g.:
#   nohup bash project_commands.sh > full_run.log 2>&1 &
# or inside `tmux`/`screen`. Every command below runs in the FOREGROUND
# (sequentially) - the outer nohup/tmux is what protects the whole run, not
# anything inside this script.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

# ============================================================
# 1. Paths - MUST match how download.txt's @PROJECT@ was actually resolved.
#    Fill in PROJECT_NAME before running.
# ============================================================
PROJECT_NAME="CHANGE_ME"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_4B="${BASE_DIR}/models/Qwen3-4B"
MODEL_8B="${BASE_DIR}/models/Qwen3-8B"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

# ============================================================
# 2. GPU topology - 2 GPU/job (1 main + 1 vLLM replica), per the agreed
#    default. Change MAIN_GPU/VLLM_GPU_IDS if the real server has a
#    different number/layout of free GPUs.
# ============================================================
MAIN_GPU=0
VLLM_GPU_IDS="1"
VLLM_BASE_PORT=8100
# Same conda env as everything else by default. On the old A100 cluster,
# vLLM needed a SEPARATE env (torch/CUDA wheel conflict) - if the same
# happens here, point this at that env's absolute vllm binary instead.
VLLM_EXECUTABLE="vllm"
# vLLM's own default is 0.9. Raise toward 0.95 if eval throughput is the
# bottleneck; B200's much larger VRAM than the A100s this project was
# originally tuned on gives real headroom here.
GPU_MEM_UTIL=0.9

CHECKPOINTS="20 25 40 50 60 75 80 100"
BENCHMARKS="aime25 aime26 hmmt25"

train() {
  local script=$1 model_path=$2 output_dir=$3
  echo "=== training: $script -> $output_dir ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --seed 0 --output-dir "${output_dir}" --skip-eval \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      > "${output_dir}_train.log" 2>&1
}

train_dry_run() {
  local script=$1 model_path=$2 output_dir=$3
  echo "=== DRY RUN (3 steps, sanity check only): $script ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --seed 0 --output-dir "${output_dir}_dryrun" --skip-eval \
      --training-steps 3 \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      > "${output_dir}_dryrun_train.log" 2>&1
  echo "Dry run finished - check ${output_dir}_dryrun_train.log for errors before continuing."
}

eval_checkpoint() {
  local script=$1 model_path=$2 output_dir=$3 tag=$4 step=$5
  local ckpt="${output_dir}/${tag}_checkpoint_step${step}"
  if [ ! -d "$ckpt" ]; then
    echo "skip (no checkpoint): $ckpt"
    return 0
  fi
  echo "=== eval: $script step ${step} ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --skip-train \
      --checkpoint-path "$ckpt" \
      --output-dir "${output_dir}" \
      --eval-engine vllm --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --eval-benchmarks ${BENCHMARKS} \
      > "${output_dir}_eval_step${step}.log" 2>&1
}

# ============================================================
# 3. Dry run first - a few training steps on the 4B/RLSD combo, to surface
#    path/GPU/env mistakes cheaply before committing to the full run below.
#    Inspect results_rlsd_4b_dryrun_train.log; only proceed once it's clean.
# ============================================================
train_dry_run run_rlsd_experiment_4b.py "${MODEL_4B}" results_rlsd_4b

# ============================================================
# 4. Train all 4 method/model combinations.
# ============================================================
train run_rlsd_experiment_4b.py "${MODEL_4B}" results_rlsd_4b
train run_sdpo_experiment_4b.py "${MODEL_4B}" results_sdpo_4b
train run_rlsd_experiment_8b.py "${MODEL_8B}" results_rlsd_8b
train run_sdpo_experiment_8b.py "${MODEL_8B}" results_sdpo_8b

# ============================================================
# 5. Eval every saved checkpoint for every combination.
# ============================================================
for step in $CHECKPOINTS; do eval_checkpoint run_rlsd_experiment_4b.py "${MODEL_4B}" results_rlsd_4b rlsd "$step"; done
for step in $CHECKPOINTS; do eval_checkpoint run_sdpo_experiment_4b.py "${MODEL_4B}" results_sdpo_4b sdpo "$step"; done
for step in $CHECKPOINTS; do eval_checkpoint run_rlsd_experiment_8b.py "${MODEL_8B}" results_rlsd_8b rlsd "$step"; done
for step in $CHECKPOINTS; do eval_checkpoint run_sdpo_experiment_8b.py "${MODEL_8B}" results_sdpo_8b sdpo "$step"; done

# ============================================================
# 6. Aggregate every avg@12/pass@12 line into one final table + JSON.
# ============================================================
python - <<'PYEOF'
import glob, json, re

pattern = re.compile(
    r"\[(?P<tag>[\w]+)_step(?P<step>\d+)\]\s+(?P<bench>aime25|aime26|hmmt25)\s+"
    r"avg@12=(?P<avg>[\d.]+)\s+pass@12=(?P<pass_>[\d.]+)"
)
rows = []
for path in sorted(glob.glob("results_*_eval_step*.log")):
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in pattern.finditer(text):
        rows.append({
            "file": path, "tag": m["tag"], "step": int(m["step"]), "benchmark": m["bench"],
            "avg@12": float(m["avg"]), "pass@12": float(m["pass_"]),
        })

with open("final_results.json", "w") as f:
    json.dump(rows, f, indent=2)

print(f"{'tag':<8} {'step':>4}  {'benchmark':<8} {'avg@12':>7} {'pass@12':>8}")
for r in rows:
    print(f"{r['tag']:<8} {r['step']:>4}  {r['benchmark']:<8} {r['avg@12']:>7.3f} {r['pass@12']:>8.3f}")
print(f"\n{len(rows)} rows written to final_results.json")
PYEOF

echo "ALL DONE - see final_results.json for the aggregated table."
