
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

source /mnt/local/uvenvs/tropic/bin/activate

# ============================================================
# 1. Paths - MUST match how download.txt's @PROJECT@ was actually resolved.
#    Fill in PROJECT_NAME before running. Dataset paths reuse the SAME
#    locations as offline_rlsd_sdpo_b200/offline_tropic_g_b200's own
#    download.txt (already downloaded there if those ran first under the
#    same @PROJECT@) - only the OLMo model itself is new.
# ============================================================
PROJECT_NAME="CHANGE_ME"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_OLMO="${BASE_DIR}/models/Olmo-3-7B-Think"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

# ============================================================
# 2. GPU topology - same as offline_rlsd_sdpo_b200/offline_tropic_g_b200,
#    confirmed with the server owner: GPU 2 (main/training) + GPU 3 (vLLM
#    replica). Change MAIN_GPU/VLLM_GPU_IDS below if that ever changes.
# ============================================================
MAIN_GPU=2
VLLM_GPU_IDS="3"
VLLM_BASE_PORT=8100
VLLM_EXECUTABLE="vllm"
GPU_MEM_UTIL=0.9

# Per-method eval cadence (checkpoints are still SAVED at all 8 steps below
# by each script's own CHECKPOINT_STEPS - this only controls which of those
# get EVALUATED, per the user's explicit choice: RLSD/SDPO (baselines) eval
# less densely than TROPIC-G (the proposal). SDPO's cadence is ASSUMED same
# as RLSD (not separately specified) - flag if that's wrong.
CHECKPOINTS_RLSD="25 50 75 100"
CHECKPOINTS_SDPO="25 50 75 100"
CHECKPOINTS_TROPIC="20 25 40 50 75 100"
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
  echo "IMPORTANT (OLMo-specific): also inspect a few RAW student rollouts in"
  echo "this dry run (not just that it doesn't crash) - the empty-think-prefill"
  echo "trick (tropic/data.py's empty_think_suffix) is UNVERIFIED end-to-end for"
  echo "Olmo-3-7B-Think; confirm the model actually skips reasoning and doesn't"
  echo "just ignore the closed empty <think></think> block before trusting a full run."
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
# 3. Dry run first - a few training steps on RLSD/OLMo, to surface
#    path/GPU/env mistakes cheaply AND to sanity-check the empty-think-
#    prefill trick before committing to the full run below. Inspect
#    results_rlsd_olmo7b_dryrun_train.log; only proceed once it's clean.
# ============================================================
train_dry_run run_rlsd_experiment_olmo7b.py "${MODEL_OLMO}" results_rlsd_olmo7b

# ============================================================
# 4. Train all 3 methods on OLMo-3-7B-Think. RLSD/SDPO/TROPIC-G already ran
#    on Qwen3-4B/8B separately (offline_rlsd_sdpo_b200/, offline_tropic_g_b200/)
#    - this folder is only the OLMo addition, not a re-run of those.
# ============================================================
train run_rlsd_experiment_olmo7b.py "${MODEL_OLMO}" results_rlsd_olmo7b
train run_sdpo_experiment_olmo7b.py "${MODEL_OLMO}" results_sdpo_olmo7b
train run_tropic_g_experiment_olmo7b.py "${MODEL_OLMO}" results_tropic_g_olmo7b
train run_tropic_g_topk64_olmo7b.py "${MODEL_OLMO}" results_tropic_g_topk64_olmo7b

# ============================================================
# 5. Eval every saved checkpoint for all 4 method/config combinations.
# ============================================================
for step in $CHECKPOINTS_RLSD; do eval_checkpoint run_rlsd_experiment_olmo7b.py "${MODEL_OLMO}" results_rlsd_olmo7b rlsd "$step"; done
for step in $CHECKPOINTS_SDPO; do eval_checkpoint run_sdpo_experiment_olmo7b.py "${MODEL_OLMO}" results_sdpo_olmo7b sdpo "$step"; done
for step in $CHECKPOINTS_TROPIC; do eval_checkpoint run_tropic_g_experiment_olmo7b.py "${MODEL_OLMO}" results_tropic_g_olmo7b tropic_g_olmo "$step"; done
for step in $CHECKPOINTS_TROPIC; do eval_checkpoint run_tropic_g_topk64_olmo7b.py "${MODEL_OLMO}" results_tropic_g_topk64_olmo7b tropic_g_topk64_olmo "$step"; done

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

print(f"{'tag':<16} {'step':>4}  {'benchmark':<8} {'avg@12':>7} {'pass@12':>8}")
for r in rows:
    print(f"{r['tag']:<16} {r['step']:>4}  {r['benchmark']:<8} {r['avg@12']:>7.3f} {r['pass@12']:>8.3f}")
print(f"\n{len(rows)} rows written to final_results.json")
PYEOF

echo "ALL DONE - see final_results.json for the aggregated table."
