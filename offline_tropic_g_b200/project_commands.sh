
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

source /mnt/local/uvenvs/tropic/bin/activate

# ============================================================
# 1. Paths - MUST match how download.txt's @PROJECT@ was actually resolved.
#    Fill in PROJECT_NAME before running. If RLSD/SDPO (offline_rlsd_sdpo_b200/)
#    already ran on this same server under the same @PROJECT@, the model/
#    dataset files below are ALREADY on disk - this download.txt targets the
#    exact same paths on purpose, so nothing needs to be re-fetched.
# ============================================================
PROJECT_NAME="CHANGE_ME"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_4B="${BASE_DIR}/models/Qwen3-4B"
MODEL_8B="${BASE_DIR}/models/Qwen3-8B"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

# ============================================================
# 2. GPU topology - same as offline_rlsd_sdpo_b200/project_commands.sh,
#    confirmed with the server owner: GPU 2 (main/training) + GPU 3 (vLLM
#    replica). Change MAIN_GPU/VLLM_GPU_IDS below if that ever changes.
# ============================================================
MAIN_GPU=2
VLLM_GPU_IDS="3"
VLLM_BASE_PORT=8100
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
      2>&1 | tee "${output_dir}_train.log"
}

train_dry_run() {
  local script=$1 model_path=$2 output_dir=$3
  echo "=== DRY RUN (3 steps, sanity check only): $script ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --seed 0 --output-dir "${output_dir}_dryrun" --skip-eval \
      --training-steps 3 \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      2>&1 | tee "${output_dir}_dryrun_train.log"
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
      2>&1 | tee "${output_dir}_eval_step${step}.log"
}

# ============================================================
# 3. Dry run first - a few training steps on the 4B combo, to surface
#    path/GPU/env mistakes cheaply before committing to the full run below.
#    Inspect results_tropic_g_4b_dryrun_train.log; only proceed once it's clean.
# ============================================================
train_dry_run run_tropic_g_experiment_4b.py "${MODEL_4B}" results_tropic_g_4b

# ============================================================
# 4. Train both model sizes. Only TROPIC-G here - RLSD/SDPO already ran
#    separately (offline_rlsd_sdpo_b200/), this package is a standalone
#    addition, not a re-run of those.
# ============================================================
train run_tropic_g_experiment_4b.py "${MODEL_4B}" results_tropic_g_4b
train run_tropic_g_experiment_8b.py "${MODEL_8B}" results_tropic_g_8b

# ============================================================
# 5. Eval every saved checkpoint for both model sizes.
# ============================================================
for step in $CHECKPOINTS; do eval_checkpoint run_tropic_g_experiment_4b.py "${MODEL_4B}" results_tropic_g_4b tropic_g "$step"; done
for step in $CHECKPOINTS; do eval_checkpoint run_tropic_g_experiment_8b.py "${MODEL_8B}" results_tropic_g_8b tropic_g "$step"; done

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
