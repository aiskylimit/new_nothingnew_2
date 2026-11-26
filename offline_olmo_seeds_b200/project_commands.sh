#!/bin/bash
# Extra TRAINING SEEDS (1 and 2) of TROPIC-G (top-64) and OPSD on OLMo-3-7B-Think, for the rebuttal appendix
# (mean +- std over seeds). Seed 0 is the paper's run and is NOT re-run here.
#
# Settings = paper Table 10, unchanged from the paper runs: effective batch 32 rollouts/step, LoRA r64/a128,
# lr 5e-6, train rollouts temperature 1.0 / top_p 1.0 / top_k -1 (<=1024 tokens), eval avg@12 with
# temperature 1.0 / top_p 0.95 up to 38912 tokens on aime25/aime26/hmmt25.
#   TROPIC: alpha 0.25, epsilon 0.1, beta 0.1, top-64; trained 100 steps (as seed 0), evaluated at 10/15/20/25/40.
#   OPSD:   frozen teacher, beta 0 (forward KL), jsd_token_clip 0.06; trained 100 steps, evaluated at 25/50/75/100.
# Same evaluated checkpoints as the paper (Table 2). TROPIC is still trained the full 100 steps like seed 0, so its
# training log gives the Figure 1b rollout-length curve up to step 100 (no extra evals).
# Training logs carry the token counters of the Qwen3-8B token-efficiency scripts (rollout tokens, rollouts,
# truncation, teacher forwards, grad norm, per-step KL drift) for Figure 1 and the cost analysis.
#
# Scheduling - keeps both GPUs busy all the time:
#   Two workers share GPUs 2 and 3 and pull jobs from ONE ordered list (dry runs, 4 trainings, 18 evals);
#   an eval becomes available as soon as its training is done. Worker 0 trains on GPU 2 with its rollout
#   server on GPU 3, worker 1 the other way round; every eval uses one vLLM replica on each GPU. So each GPU
#   always hosts at most one process per worker, and while one eval drains its few longest generations the
#   other worker's job keeps the GPU busy. Each vLLM instance gets GPU_MEM_UTIL = 0.45 of a B200 (~80 GB,
#   enough KV cache for many 41k-token sequences); a trainer needs well under the remaining ~90 GB.
#   A failed job is retried once (e.g. two vLLM servers starting at the same moment); finished jobs are
#   skipped, so the script can simply be re-submitted after a crash.
#
# At the end, still on this server: every generation is re-graded and reduced to per-sample records
# (rebuttal/import_legacy.py -> runs/olmo3-7b-think/<method>/seed<k>/, a few MB instead of ~1 GB of text),
# rebuttal/analyze.py writes the seed tables (analysis/report.md, appendix_seeds_table.tex, CSVs) and
# rebuttal/plots.py the figures when matplotlib is installed. Everything needed later - runs/, analysis/,
# final_results.json, the tails of all logs - goes into ONE small file, olmo_seeds_results.tgz (split into
# 24 MB parts olmo_seeds_results.tgz.part-* if it is larger). Generations and LoRA checkpoints stay on the server.
# Locally: tar xzf olmo_seeds_results.tgz (or: cat olmo_seeds_results.tgz.part-* | tar xzf -), copy runs/ next to
# the seed-0 imports and re-run rebuttal/analyze.py + plots.py for the tables and figures over seeds 0-2.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH=.

source /mnt/local/uvenvs/tropic/bin/activate

# ---- paths (same @PROJECT@ as offline_olmo7b_b200/, so the model and data are already on disk) ----
PROJECT_NAME="aiskylimit_new_nothingnew_2"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_OLMO="${BASE_DIR}/models/Olmo-3-7B-Think"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

# ---- what to run ----
GPU_A=2; GPU_B=3
WORKERS=2                 # 1 = a single job at a time (then set GPU_MEM_UTIL=0.95)
GPU_MEM_UTIL=0.45         # per vLLM instance; two instances (one per worker) may share a GPU
SEEDS="1 2"
DRY_RUN=1                 # 3-step sanity run of each method before its real trainings
VLLM_EXECUTABLE="vllm"
VLLM_MAX_MODEL_LEN=41000
BENCHMARKS="aime25 aime26 hmmt25"
MAX_TRIES=2
STAGGER=${STAGGER:-180}; RETRY_WAIT=${RETRY_WAIT:-120}; POLL=${POLL:-60}   # seconds
STATE="$ROOT/.queue_state"

cfg() {  # method -> script, tag, training steps, evaluated steps
  case $1 in
    tropic) SCRIPT=run_tropic_g_topk64_olmo7b.py; TAG=tropic_g_topk64_olmo; TRAIN_STEPS=100
            CORE_STEPS="10 15 20 25 40"; LATE_STEPS="" ;;
    opsd)   SCRIPT=run_opsd_experiment_olmo7b.py; TAG=opsd_olmo;            TRAIN_STEPS=100
            CORE_STEPS="25 50 75 100";   LATE_STEPS="" ;;
  esac
  STEPS="$CORE_STEPS $LATE_STEPS"
}
MODEL_KEY=olmo3-7b-think
out_dir() { echo "results_olmo7b_$1_seed$2"; }

has_eval() {  # has_eval <dir> <tag> <step>   (no pipeline: pipefail would turn a missing glob into "no")
  local f
  for f in "$1/$2_step$3"_*generations.json "$1/$2_step$3_generations.json"; do
    [[ -e "$f" && "$f" != *_shard* ]] && return 0
  done
  return 1
}

# ---- job list, in priority order: Table 2 checkpoints (seed 1 first), then the Figure 1-only checkpoints ----
JOBS=()
if [ "$DRY_RUN" = "1" ]; then JOBS+=(dry:tropic dry:opsd); fi
for s in $SEEDS; do JOBS+=(train:tropic:$s train:opsd:$s); done
for s in $SEEDS; do
  for m in tropic opsd; do cfg $m; for k in $CORE_STEPS; do JOBS+=(eval:$m:$s:$k); done; done
done
for s in $SEEDS; do
  for m in tropic opsd; do cfg $m; for k in $LATE_STEPS; do JOBS+=(eval:$m:$s:$k); done; done
done

job_done() {
  IFS=: read -r kind m s k <<< "$1"; cfg "$m"
  case $kind in
    dry)   [ -f "$STATE/ok_dry_$m" ] || [ -f "$(out_dir $m ${SEEDS%% *})/${TAG}_checkpoint_step${TRAIN_STEPS}/adapter_config.json" ] ;;
    train) [ -f "$(out_dir $m $s)/${TAG}_checkpoint_step${TRAIN_STEPS}/adapter_config.json" ] ;;
    eval)  has_eval "$(out_dir $m $s)" "$TAG" "$k" ;;
  esac
}
job_dep() {  # the job this one waits for (empty = none)
  IFS=: read -r kind m s k <<< "$1"
  case $kind in
    train) [ "$DRY_RUN" = "1" ] && echo "dry:$m" ;;
    eval)  echo "train:$m:$s" ;;
  esac
}
gave_up() { [ -f "$STATE/gaveup_${1//:/_}" ]; }

run_job() {  # run_job <job> <worker>
  IFS=: read -r kind m s k <<< "$1"; cfg "$m"
  local w=$2 main repl port log out
  if [ "$w" = "0" ]; then main=$GPU_A; repl=$GPU_B; else main=$GPU_B; repl=$GPU_A; fi
  port=$((8100 + 100 * w))
  local common=(--model-path "$MODEL_OLMO" --vllm-base-port "$port" --vllm-executable "$VLLM_EXECUTABLE"
                --vllm-gpu-memory-utilization "$GPU_MEM_UTIL" --vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
  case $kind in
    dry)
      out="results_olmo7b_${m}_dryrun"; log="${out}_train.log"; rm -rf "$out"
      env CUDA_VISIBLE_DEVICES=$main python "scripts/$SCRIPT" "${common[@]}" --seed 0 --output-dir "$out" \
          --skip-eval --training-steps 3 --use-vllm-rollout --vllm-gpu-ids "$repl" > "$log" 2>&1
      [ -f "$out/${TAG}_train_logs.json" ] && touch "$STATE/ok_dry_$m" ;;
    train)
      out=$(out_dir $m $s); log="${out}_train.log"
      env CUDA_VISIBLE_DEVICES=$main python "scripts/$SCRIPT" "${common[@]}" --seed "$s" --output-dir "$out" \
          --skip-eval --training-steps "$TRAIN_STEPS" --use-vllm-rollout --vllm-gpu-ids "$repl" > "$log" 2>&1 ;;
    eval)
      out=$(out_dir $m $s); log="${out}_eval_step${k}.log"
      env CUDA_VISIBLE_DEVICES=$main python "scripts/$SCRIPT" "${common[@]}" --skip-train \
          --checkpoint-path "$ROOT/$out/${TAG}_checkpoint_step${k}" --output-dir "$ROOT/$out" \
          --eval-engine vllm --vllm-gpu-ids "$GPU_A,$GPU_B" --eval-benchmarks $BENCHMARKS > "$log" 2>&1 ;;
  esac
  job_done "$1" && return 0
  echo "  (log: $log)"; return 1
}

worker() {  # worker <id>
  local w=$1 job dep n waiting ran
  [ "$w" != "0" ] && sleep "$STAGGER"  # do not start two vLLM servers on the same GPU at the same moment
  while true; do
    waiting=0; ran=0
    for job in "${JOBS[@]}"; do
      job_done "$job" && continue
      gave_up "$job" && continue
      dep=$(job_dep "$job")
      if [ -n "$dep" ] && ! job_done "$dep"; then
        if gave_up "$dep"; then
          [ -f "$STATE/gaveup_${job//:/_}" ] || { echo "[w$w] skip $job: $dep failed"; touch "$STATE/gaveup_${job//:/_}"; }
        else waiting=1; fi
        continue
      fi
      mkdir "$STATE/claim_${job//:/_}" 2>/dev/null || continue      # someone else is running it
      n=$(( $(cat "$STATE/tries_${job//:/_}" 2>/dev/null || echo 0) + 1 )); echo $n > "$STATE/tries_${job//:/_}"
      echo "[w$w] START $job (try $n) $(date '+%F %T')"
      if run_job "$job" "$w"; then
        echo "[w$w] done  $job $(date '+%F %T')"
      else
        echo "[w$w] !!! FAILED $job (try $n) $(date '+%F %T')"
        if [ "$n" -ge "$MAX_TRIES" ]; then touch "$STATE/gaveup_${job//:/_}"; else sleep "$RETRY_WAIT"; fi
      fi
      rmdir "$STATE/claim_${job//:/_}"
      ran=1; break                 # rescan from the top, so jobs are always taken in priority order
    done
    if [ "$ran" = "0" ]; then
      [ "$waiting" = "1" ] || break
      sleep "$POLL"
    fi
  done
  echo "[w$w] no jobs left $(date '+%F %T')"
}

rm -rf "$STATE"; mkdir -p "$STATE"
echo "=== OLMo seeds $SEEDS on GPUs $GPU_A,$GPU_B: ${#JOBS[@]} jobs, $WORKERS worker(s) ($(date)) ==="
for ((w = 0; w < WORKERS; w++)); do worker $w & done
wait

echo "=== summary ==="
for job in "${JOBS[@]}"; do job_done "$job" && echo "  ok      $job" || echo "  MISSING $job"; done

# ---- aggregate every finished eval into final_results.json ----
python - <<'PYEOF'
import glob, json, re
pat = re.compile(r"\[(?P<tag>[A-Za-z0-9_]+?)_step(?P<step>\d+)[^\]]*\]\s+(?P<bench>aime25|aime26|hmmt25)\s+"
                 r"avg@12=(?P<avg>[\d.]+)\s+pass@12=(?P<pass_>[\d.]+)")
rows = []
for path in sorted(glob.glob("results_olmo7b_*_seed*_eval_step*.log")):
    m0 = re.match(r"results_olmo7b_(?P<method>\w+?)_seed(?P<seed>\d+)_eval_step\d+\.log", path)
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in pat.finditer(text):
        rows.append(dict(method=m0["method"], seed=int(m0["seed"]), step=int(m["step"]), benchmark=m["bench"],
                         avg12=float(m["avg"]), pass12=float(m["pass_"]), log=path))
json.dump(rows, open("final_results.json", "w"), indent=2)
print(f"{'method':<7} {'seed':>4} {'step':>4}  {'benchmark':<8} {'avg@12':>7} {'pass@12':>8}")
for r in sorted(rows, key=lambda r: (r["method"], r["seed"], r["step"], r["benchmark"])):
    print(f"{r['method']:<7} {r['seed']:>4} {r['step']:>4}  {r['benchmark']:<8} {r['avg12']:>7.3f} {r['pass12']:>8.3f}")
print(f"\n{len(rows)} rows written to final_results.json")
PYEOF

# ---- re-grade every generation into compact per-sample records (runs/), then the seed analysis ----
for m in tropic opsd; do
  cfg $m
  for s in $SEEDS; do
    out=$(out_dir $m $s)
    [ -d "$out" ] || continue
    echo "=== import $m seed $s ==="
    python rebuttal/import_legacy.py --src "$out" --tag "$TAG" --model "$MODEL_KEY" --method "$m" --seed "$s" \
        --steps $STEPS --tokenizer "$MODEL_OLMO" --workers 16 --overwrite \
      || echo "!!! import failed for $out (its generations are still in $out/)"
  done
done
python rebuttal/analyze.py --runs-root runs --out analysis || echo "!!! analyze failed"
if python -c "import matplotlib" 2>/dev/null; then
  python rebuttal/plots.py --analysis analysis || echo "!!! plots failed"
else
  echo "(matplotlib not installed - figures will be made locally from runs/)"
fi

# ---- ONE small bundle: compact results + analysis + log tails (no generations, no checkpoints) ----
mkdir -p log_tails
for f in results_olmo7b_*_*.log; do [ -e "$f" ] && tail -n 300 "$f" > "log_tails/$f"; done
for s in $SEEDS; do for m in tropic opsd; do cfg $m; out=$(out_dir $m $s)
  [ -f "$out/${TAG}_train_logs.json" ] && cp "$out/${TAG}_train_logs.json" "log_tails/${out}_train_logs.json"
done; done
rm -f olmo_seeds_results.tgz olmo_seeds_results.tgz.part-*
tar czf olmo_seeds_results.tgz final_results.json runs analysis log_tails 2>/dev/null
size=$(stat -c %s olmo_seeds_results.tgz)
if [ "$size" -gt $((24 * 1024 * 1024)) ]; then
  split -b 24M olmo_seeds_results.tgz olmo_seeds_results.tgz.part-
  echo "bundle is $((size / 1024 / 1024)) MB - split into: $(ls olmo_seeds_results.tgz.part-* | tr '\n' ' ')"
fi
echo "ALL DONE - fetch olmo_seeds_results.tgz ($(du -h olmo_seeds_results.tgz | cut -f1)); report: analysis/report.md"
[ -f analysis/report.md ] && head -n 80 analysis/report.md
