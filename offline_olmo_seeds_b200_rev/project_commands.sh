#!/bin/bash
# SECOND RUNNER for the OLMo-3-7B-Think seeds 1-2 evals, on GPUs 4,5, next to the run already going on GPUs 2,3
# (offline_olmo_seeds_b200/). It evaluates the SAME checkpoints, writes into the SAME result folders, and takes
# the eval jobs from the END of the list (OPSD seed 2 step 100 first), while the first run works from the top -
# the two meet in the middle, nothing is evaluated twice:
#   * both runners claim a job with the same lock (offline_olmo_seeds_b200/.queue_state/claim_<job>), so a job
#     one of them is running is skipped by the other;
#   * a job counts as done only when ALL three benchmarks are graded (<tag>_step<k>*_eval_results.json), so an
#     eval that died half-way is redone here;
#   * after each finished eval the shared bundle olmo_seeds_results.tgz is refreshed (same lock as the first run).
# Same code, model, data and settings as the first run (scripts/, tropic/, rebuttal/ of offline_olmo_seeds_b200/);
# only the GPUs (4,5), the vLLM ports (8300+) and the job order differ. Nothing is trained here.
# Fetch the results as before:  #2 -f-/mnt/local/aiskylimit_new_nothingnew_2/offline_olmo_seeds_b200/olmo_seeds_results.tgz
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../offline_olmo_seeds_b200" || { echo "!!! no offline_olmo_seeds_b200/ next to $HERE"; exit 1; }
ROOT="$(pwd)"
export PYTHONPATH=.

VENV=/mnt/local/uvenvs/tropic            # built by a "#i tropic.txt" commit
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; else echo "!!! no venv at $VENV"; fi

# ---- what to run ----
GPU_A=${GPU_A:-4}; GPU_B=${GPU_B:-5}
WORKERS=2                 # 1 = a single job at a time (then set GPU_MEM_UTIL=0.95)
GPU_MEM_UTIL=0.45         # per vLLM instance; two instances (one per worker) may share a GPU
SEEDS="1 2"
DRY_RUN=0                 # evals only: the checkpoints come from the first run
VLLM_EXECUTABLE="vllm"
VLLM_MAX_MODEL_LEN=41000
BENCHMARKS="aime25 aime26 hmmt25"
MAX_TRIES=2
STAGGER=${STAGGER:-180}; RETRY_WAIT=${RETRY_WAIT:-120}; POLL=${POLL:-60}   # seconds
STATE="$ROOT/.queue_state"          # SHARED with the first run (claims, packer lock)

# ---- model + data: the same files may sit in several places on the server, depending on which download
#      ("#d") commit fetched them - this folder's download.txt (tropic_baselines/, as for offline_olmo7b_b200),
#      opsd/download.txt (OPSD/models, OPSD/data/raw), or the newer /mnt/local/_data/ layout. Take the first
#      complete copy; as a last resort search /mnt/local. Nothing found = the preflight below says so. ----
PROJECT_NAME="aiskylimit_new_nothingnew_2"
ROOTS="/mnt/local/${PROJECT_NAME} /mnt/local/_data/${PROJECT_NAME} /mnt/local/_data"
model_ok() { [ -f "$1/config.json" ] && ls "$1"/*.safetensors > /dev/null 2>&1; }
data_ok() {  # <dir holding train/ and eval/<bench>/>
  local b
  [ -n "$(ls -A "$1/train" 2>/dev/null)" ] || return 1
  for b in $BENCHMARKS; do [ -n "$(ls -A "$1/eval/$b" 2>/dev/null)" ] || return 1; done
}
MODEL_OLMO=""; DATA_DIR=""
for r in $ROOTS; do
  for sub in tropic_baselines/models OPSD/models models; do
    if [ -z "$MODEL_OLMO" ] && model_ok "$r/$sub/Olmo-3-7B-Think"; then MODEL_OLMO="$r/$sub/Olmo-3-7B-Think"; fi
  done
  for sub in tropic_baselines/data OPSD/data/raw data; do
    if [ -z "$DATA_DIR" ] && data_ok "$r/$sub"; then DATA_DIR="$r/$sub"; fi
  done
done
if [ -z "$MODEL_OLMO" ]; then
  while IFS= read -r c; do model_ok "$c" && { MODEL_OLMO=$c; break; }; done \
    < <(find /mnt/local -maxdepth 6 -type d -name "Olmo-3-7B-Think" -not -path "*/uvenvs/*" 2>/dev/null)
fi
if [ -z "$DATA_DIR" ]; then
  while IFS= read -r c; do c=$(dirname "$(dirname "$c")"); data_ok "$c" && { DATA_DIR=$c; break; }; done \
    < <(find /mnt/local -maxdepth 7 -type d -name "aime26" -path "*/eval/aime26" -not -path "*/uvenvs/*" 2>/dev/null)
fi
# defaults only matter for the error message when nothing was found
MODEL_OLMO=${MODEL_OLMO:-/mnt/local/${PROJECT_NAME}/tropic_baselines/models/Olmo-3-7B-Think}
DATA_DIR=${DATA_DIR:-/mnt/local/${PROJECT_NAME}/tropic_baselines/data}
export TROPIC_TRAIN_DATA_PATH="${DATA_DIR}/train"
export TROPIC_EVAL_DATA_DIR="${DATA_DIR}/eval"

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

eval_complete() {  # eval_complete <dir> <tag> <step>: every benchmark of BENCHMARKS in its eval_results.json
  local f b
  for f in "$1/$2_step$3"_*eval_results.json "$1/$2_step$3_eval_results.json"; do
    [[ -e "$f" && "$f" != *_shard* ]] || continue
    for b in $BENCHMARKS; do grep -q "\"$b\"" "$f" || continue 2; done
    return 0
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
# this runner: the eval jobs only, in REVERSE order (the first run takes them from the top)
REV=()
for ((i = ${#JOBS[@]} - 1; i >= 0; i--)); do [[ "${JOBS[$i]}" == eval:* ]] && REV+=("${JOBS[$i]}"); done
JOBS=("${REV[@]}")

job_done() {
  IFS=: read -r kind m s k <<< "$1"; cfg "$m"
  case $kind in
    dry)   [ -f "$STATE/ok_dry_$m" ] || [ -f "$(out_dir $m ${SEEDS%% *})/${TAG}_checkpoint_step${TRAIN_STEPS}/adapter_config.json" ] ;;
    train) [ -f "$(out_dir $m $s)/${TAG}_checkpoint_step${TRAIN_STEPS}/adapter_config.json" ] ;;
    eval)  eval_complete "$(out_dir $m $s)" "$TAG" "$k" ;;
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

# ---- progress in the main output: a training line every PROGRESS_EVERY steps, each benchmark score as soon as
#      it is graded, and the scores of the checkpoint on its "done" line. The full logs stay in the job log files.
PROGRESS_EVERY=${PROGRESS_EVERY:-10}
TAIL_PID_OK=0; tail --pid=$$ -n 0 /dev/null > /dev/null 2>&1 && TAIL_PID_OK=1   # GNU tail; else: no live lines
progress() {  # progress <prefix> <training steps>   (stdin = a job log)
  awk -v pre="$1" -v total="$2" -v every="$PROGRESS_EVERY" '
    function num(key,   s) {
      if (!match($0, "\"" key "\": [-+0-9.eEaNinf]+")) return ""
      s = substr($0, RSTART, RLENGTH); sub(/^[^:]*: /, "", s); return s
    }
    /"student_updates_total"/ {
      st = num("step") + 1
      if (st == 1 || st % every == 0 || st == total) {
        printf "%s step %3d/%d  loss %.4g  rollout len %.0f  truncated %.2f  grad norm %.3g  drift %.3g  eta %sm\n",
               pre, st, total, num("loss"), num("mean_rollout_len"), num("rollout_truncated_frac"),
               num("grad_norm"), num("kl_student_old_drift"), num("eta_min")
        fflush()
      }
      next
    }
    / avg@[0-9]+=[0-9.]+ pass@[0-9]+=[0-9.]+$/ {
      split($(NF-1), a, "="); split($NF, p, "=")
      printf "%s %-7s %s %5.1f   %s %5.1f\n", pre, $(NF-2), a[1], 100 * a[2], p[1], 100 * p[2]
      fflush()
    }'
}
logged() {  # logged <log> <prefix> <training steps> <command...> : command -> <log>, progress lines -> stdout
  local log=$1 pre=$2 total=$3 pid; shift 3
  : > "$log"
  "$@" > "$log" 2>&1 &
  pid=$!
  if [ "$TAIL_PID_OK" = "1" ]; then
    tail --pid="$pid" -n +1 -F "$log" 2>/dev/null | progress "$pre" "$total"
  fi
  wait "$pid"
}
eval_scores() {  # eval_scores <eval log> -> "aime25 43.3 | aime26 40.1 | hmmt25 25.0" (avg@12, %)
  awk '/ avg@[0-9]+=[0-9.]+ pass@[0-9]+=[0-9.]+$/ {
         split($(NF-1), a, "="); s = s (s == "" ? "" : " | ") $(NF-2) " " sprintf("%.1f", 100 * a[2]) }
       END { print s }' "$1" 2>/dev/null
}
job_note() {  # extra text for a job's done/summary line: duration (given a start time) and eval scores
  local kind m s k t
  IFS=: read -r kind m s k <<< "$1"
  [ -n "${2:-}" ] && { t=$(( $(date +%s) - $2 )); printf " (%dh%02dm)" $((t / 3600)) $((t % 3600 / 60)); }
  [ "$kind" = "eval" ] && printf " avg@12: %s" "$(eval_scores "$(out_dir "$m" "$s")_eval_step${k}.log")"
  return 0
}

run_job() {  # run_job <job> <worker>
  IFS=: read -r kind m s k <<< "$1"; cfg "$m"
  local w=$2 main repl port log out
  if [ "$w" = "0" ]; then main=$GPU_A; repl=$GPU_B; else main=$GPU_B; repl=$GPU_A; fi
  port=$((8300 + 100 * w))
  local common=(--model-path "$MODEL_OLMO" --vllm-base-port "$port" --vllm-executable "$VLLM_EXECUTABLE"
                --vllm-gpu-memory-utilization "$GPU_MEM_UTIL" --vllm-max-model-len "$VLLM_MAX_MODEL_LEN")
  case $kind in
    dry)
      out="results_olmo7b_${m}_dryrun"; log="${out}_train.log"; rm -rf "$out"
      logged "$log" "[r$w]   $m dry" 3 \
        env CUDA_VISIBLE_DEVICES=$main python "scripts/$SCRIPT" "${common[@]}" --seed 0 --output-dir "$out" \
          --skip-eval --training-steps 3 --use-vllm-rollout --vllm-gpu-ids "$repl"
      [ -f "$out/${TAG}_train_logs.json" ] && touch "$STATE/ok_dry_$m" ;;
    train)
      out=$(out_dir $m $s); log="${out}_train.log"
      logged "$log" "[r$w]   $m s$s" "$TRAIN_STEPS" \
        env CUDA_VISIBLE_DEVICES=$main python "scripts/$SCRIPT" "${common[@]}" --seed "$s" --output-dir "$out" \
          --skip-eval --training-steps "$TRAIN_STEPS" --use-vllm-rollout --vllm-gpu-ids "$repl" ;;
    eval)
      out=$(out_dir $m $s); log="${out}_eval_step${k}.log"
      logged "$log" "[r$w]   $m s$s step$k" 0 \
        env CUDA_VISIBLE_DEVICES=$main python "scripts/$SCRIPT" "${common[@]}" --skip-train \
          --checkpoint-path "$ROOT/$out/${TAG}_checkpoint_step${k}" --output-dir "$ROOT/$out" \
          --eval-engine vllm --vllm-gpu-ids "$GPU_A,$GPU_B" --eval-benchmarks $BENCHMARKS ;;
  esac
  job_done "$1" && return 0
  echo "  (log: $log) - last lines:"
  tail -n 40 "$log" 2>/dev/null | sed 's/^/  | /'     # the error itself, right in the main output
  for f in "$out"/vllm*.log; do
    [ -e "$f" ] && { echo "  ($f) - last lines:"; tail -n 15 "$f" | sed 's/^/  | /'; }
  done
  return 1
}

worker() {  # worker <id>
  local w=$1 job dep n waiting ran t0
  [ "$w" != "0" ] && sleep "$STAGGER"  # do not start two vLLM servers on the same GPU at the same moment
  while true; do
    waiting=0; ran=0
    for job in "${JOBS[@]}"; do
      job_done "$job" && continue
      gave_up "$job" && continue
      dep=$(job_dep "$job")
      if [ -n "$dep" ] && ! job_done "$dep"; then
        if gave_up "$dep"; then
          [ -f "$STATE/gaveup_${job//:/_}" ] || { echo "[r$w] skip $job: $dep failed"; touch "$STATE/gaveup_${job//:/_}"; }
        else waiting=1; fi
        continue
      fi
      mkdir "$STATE/claim_${job//:/_}" 2>/dev/null || continue      # the other runner (or worker) is on it
      touch "$HELD/claim_${job//:/_}"
      n=$(( $(cat "$STATE/tries_${job//:/_}" 2>/dev/null || echo 0) + 1 )); echo $n > "$STATE/tries_${job//:/_}"
      echo "[r$w] START $job (try $n) $(date '+%F %T')"
      t0=$(date +%s)
      if run_job "$job" "$w"; then
        echo "[r$w] done  $job $(date '+%F %T')$(job_note "$job" "$t0")"
        # refresh the bundle in the background (the worker goes straight on to its next job), so that
        # olmo_seeds_results.tgz always holds every finished checkpoint even if the run is cut off
        [[ "$job" != dry:* ]] && { package_results incremental >> pack.log 2>&1 & }
      else
        echo "[r$w] !!! FAILED $job (try $n) $(date '+%F %T')"
        if [ "$n" -ge "$MAX_TRIES" ]; then touch "$STATE/gaveup_${job//:/_}"; else sleep "$RETRY_WAIT"; fi
      fi
      rmdir "$STATE/claim_${job//:/_}"; rm -f "$HELD/claim_${job//:/_}"
      ran=1; break                 # rescan from the top, so jobs are always taken in priority order
    done
    if [ "$ran" = "0" ]; then
      [ "$waiting" = "1" ] || break
      sleep "$POLL"
    fi
  done
  wait                             # this worker's background packers
  echo "[r$w] no jobs left $(date '+%F %T')"
}

# ---- results bundle: rebuilt after every finished train/eval job and once more at the very end ----
# package_results incremental : only newly finished evals are re-graded (a minute or so), output -> pack.log
# package_results final       : everything re-graded from scratch (--overwrite), as the original end-of-run step
package_results() {
  local mode=$1 waited=0 ow=""
  [ "$mode" = "final" ] && ow="--overwrite"
  until mkdir "$STATE/pack_lock" 2>/dev/null; do     # one packer at a time
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge 3600 ]; then echo "!!! packer lock held for 1 h - skipping this refresh"; return 1; fi
  done
  echo "=== package ($mode) $(date '+%F %T') ==="
  _package_body "$ow"
  rmdir "$STATE/pack_lock"
}

_package_body() {
local ow=$1
# ---- aggregate every finished eval into final_results.json ----
python - <<'PYEOF'
import glob, json, os, re
pat = re.compile(r"\[(?P<tag>[A-Za-z0-9_]+?)_step(?P<step>\d+)[^\]]*\]\s+(?P<bench>aime25|aime26|hmmt25)\s+"
                 r"avg@12=(?P<avg>[\d.]+)\s+pass@12=(?P<pass_>[\d.]+)")
rows = []
for path in sorted(glob.glob("results_olmo7b_*_seed*_eval_step*.log")):
    m0 = re.match(r"results_olmo7b_(?P<method>\w+?)_seed(?P<seed>\d+)_eval_step\d+\.log", path)
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in pat.finditer(text):
        rows.append(dict(method=m0["method"], seed=int(m0["seed"]), step=int(m["step"]), benchmark=m["bench"],
                         avg12=float(m["avg"]), pass12=float(m["pass_"]), log=path))
json.dump(rows, open("final_results.json.tmp", "w"), indent=2)
os.replace("final_results.json.tmp", "final_results.json")
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
    compgen -G "$out/${TAG}_step*generations.json" > /dev/null || continue   # no finished eval yet
    echo "=== import $m seed $s ==="
    python rebuttal/import_legacy.py --src "$out" --tag "$TAG" --model "$MODEL_KEY" --method "$m" --seed "$s" \
        --steps $STEPS --tokenizer "$MODEL_OLMO" --workers 16 $ow \
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
for f in results_olmo7b_*_*.log pack.log; do [ -e "$f" ] && tail -n 300 "$f" > "log_tails/$f"; done
for s in $SEEDS; do for m in tropic opsd; do cfg $m; out=$(out_dir $m $s)
  [ -f "$out/${TAG}_train_logs.json" ] && cp "$out/${TAG}_train_logs.json" "log_tails/${out}_train_logs.json"
done; done
tar czf olmo_seeds_results.tgz.tmp final_results.json runs analysis log_tails 2>/dev/null
rm -f olmo_seeds_results.tgz.part-*
mv -f olmo_seeds_results.tgz.tmp olmo_seeds_results.tgz   # swapped in whole: a fetch never sees half a file
size=$(stat -c %s olmo_seeds_results.tgz)
if [ "$size" -gt $((24 * 1024 * 1024)) ]; then
  split -b 24M olmo_seeds_results.tgz olmo_seeds_results.tgz.part-
  echo "bundle is $((size / 1024 / 1024)) MB - split into: $(ls olmo_seeds_results.tgz.part-* | tr '\n' ' ')"
fi
echo "bundle refreshed $(date '+%F %T'): $(du -h olmo_seeds_results.tgz | cut -f1)"
}

# ---- preflight: checks everything the jobs need and prints it to the main output; stops early with the
#      reason instead of letting every job die a few seconds after it starts ----
preflight() {
  local bad=0 d
  echo "=== preflight $(date '+%F %T') on $(hostname) ==="
  echo "  python: $(command -v python)   vllm: $(command -v "$VLLM_EXECUTABLE" || echo NOT FOUND)"
  [ "$(command -v python)" = "$VENV/bin/python" ] || { echo "!!! the venv $VENV is not active - (re)install it with a '#i tropic.txt' commit"; bad=1; }
  command -v "$VLLM_EXECUTABLE" > /dev/null || { echo "!!! '$VLLM_EXECUTABLE' is not on PATH"; bad=1; }
  echo "  model: $MODEL_OLMO"
  echo "  data:  $DATA_DIR"
  model_ok "$MODEL_OLMO" || { echo "!!! no complete Olmo-3-7B-Think (config.json + *.safetensors) found under /mnt/local"; bad=1; }
  data_ok "$DATA_DIR" || { echo "!!! no complete train + eval/{${BENCHMARKS// /,}} data found under /mnt/local"; bad=1; }
  if [ "$bad" = "1" ]; then
    echo "  (missing model/data -> commit the contents of offline_olmo_seeds_b200/download.txt as commands.sh ('#d') first)"
    echo "  what is on disk:"
    ls -la /mnt/local "/mnt/local/${PROJECT_NAME}" "/mnt/local/${PROJECT_NAME}/tropic_baselines" \
           "/mnt/local/${PROJECT_NAME}/tropic_baselines/models" "/mnt/local/${PROJECT_NAME}/OPSD/models" \
           "/mnt/local/_data/${PROJECT_NAME}" /mnt/local/uvenvs 2>&1 | sed 's/^/  | /' | head -80
  fi
  # GPUs 2,3: who else is on them (a job of someone else there = out-of-memory for ours)
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/  gpu /'
  local used
  for d in $GPU_A $GPU_B; do
    used=$(nvidia-smi -i "$d" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -z "$used" ]; then echo "!!! nvidia-smi cannot see GPU $d"; bad=1
    elif [ "$used" -gt 20000 ]; then echo "!!! GPU $d already has ${used} MiB in use by another process - our jobs may run out of memory"; fi
  done
  # the venv itself: imports, CUDA, and that this transformers can read the OLMo-3 config (needs >= 4.57)
  CUDA_VISIBLE_DEVICES="$GPU_A,$GPU_B" MODEL_OLMO="$MODEL_OLMO" BENCHMARKS="$BENCHMARKS" \
    python - <<'PY' 2>&1 | sed 's/^/  /' || bad=1
import importlib, os, sys
ok = True
for m in ["torch", "transformers", "peft", "math_verify", "vllm", "numpy", "pyarrow", "yaml"]:
    try:
        print(f"{m:<12} {getattr(importlib.import_module(m), '__version__', 'ok')}")
    except Exception as e:
        print(f"!!! cannot import {m}: {type(e).__name__}: {e}"); ok = False
# `datasets` (via pandas) is optional: tropic/local_dataset.py reads the local snapshots with pyarrow when it
# does not import - e.g. pandas 3.x (needs numpy >= 2) next to tropic.txt's numpy 1.26.4.
for m in ["pandas", "datasets"]:
    try:
        print(f"{m:<12} {getattr(importlib.import_module(m), '__version__', 'ok')}")
    except Exception as e:
        print(f"(note) {m} does not import ({type(e).__name__}: {str(e)[:70]}...) - data is read with pyarrow instead")
try:
    import torch
    n = torch.cuda.device_count()
    print(f"CUDA available={torch.cuda.is_available()} devices={n} " + ", ".join(torch.cuda.get_device_name(i) for i in range(n)))
    if n < 2:
        print("!!! fewer than 2 CUDA devices visible"); ok = False
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i); print(f"  device {i}: {free / 2**30:.0f} / {total / 2**30:.0f} GiB free")
except Exception as e:
    print(f"!!! CUDA init failed: {type(e).__name__}: {e}"); ok = False
tok = None
try:
    from transformers import AutoConfig, AutoTokenizer
    c = AutoConfig.from_pretrained(os.environ["MODEL_OLMO"]); tok = AutoTokenizer.from_pretrained(os.environ["MODEL_OLMO"])
    print(f"model config + tokenizer OK ({c.model_type})")
except Exception as e:
    print(f"!!! cannot load the model config/tokenizer: {type(e).__name__}: {e}"); ok = False
# the very same loaders (and arguments) the jobs call first - what a job would hit in its first seconds
try:
    from tropic.data import load_opsd_math_examples
    if tok is not None:
        ex = load_opsd_math_examples(tok, num_examples=4, only_correct=True, teacher_thinking=True,
                                     student_thinking=False, seed=0, max_length=20000,
                                     empty_think_suffix="\n\n</think>\n\n")
        print(f"training data OK ({len(ex)} examples through tropic.data)")
except Exception as e:
    print(f"!!! training data does not load: {type(e).__name__}: {e}"); ok = False
try:
    from tropic.eval import load_benchmark
    for b in os.environ["BENCHMARKS"].split():
        print(f"eval {b}: {len(load_benchmark(b, num_problems=30))} problems")
except Exception as e:
    print(f"!!! eval data does not load: {type(e).__name__}: {e}"); ok = False
sys.exit(0 if ok else 1)
PY
  # the first run's checkpoints and queue
  echo "  results folder: $ROOT"
  local job todo=0 cnt_done=0 cnt_run=0 ck
  for job in "${JOBS[@]}"; do
    IFS=: read -r kind m s k <<< "$job"; cfg "$m"
    ck="$(out_dir $m $s)/${TAG}_checkpoint_step${k}/adapter_config.json"
    if job_done "$job"; then cnt_done=$((cnt_done + 1)); st="done$(job_note "$job")"
    elif [ -d "$STATE/claim_${job//:/_}" ]; then cnt_run=$((cnt_run + 1)); st="RUNNING in the other runner"
    elif [ -f "$ck" ]; then todo=$((todo + 1)); st="to do"
    else st="!!! checkpoint missing: $ck"; bad=1; fi
    printf "    %-18s %s\n" "$job" "$st"
  done
  echo "  evals: $cnt_done done, $cnt_run running in the other runner, $todo left (this runner starts from the bottom)"
  [ "$todo" = "0" ] && echo "  (nothing left to do here)"
  if [ "$bad" = "1" ]; then echo "=== preflight FAILED - nothing was run (see the !!! lines above) ==="; return 1; fi
  echo "=== preflight OK ==="
}
preflight || exit 1

mkdir -p "$STATE"                 # NOT wiped: it holds the first run's claims
HELD="$STATE/held_rev_$$"; mkdir -p "$HELD"
release_claims() {                # stopped (time limit, kill): give our unfinished jobs back to the first run
  local c
  for c in "$HELD"/claim_*; do [ -e "$c" ] && rmdir "$STATE/$(basename "$c")" 2>/dev/null; rm -f "$c"; done
  rmdir "$HELD" 2>/dev/null
}
trap 'release_claims' EXIT
kill_tree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null; }
# stopped: give the claims back, then stop OUR workers, evals and vLLM servers only (not the caller's shell)
trap 'release_claims; trap - INT TERM; for c in $(jobs -p) $(pgrep -P $$ 2>/dev/null); do kill_tree "$c"; done; release_claims; exit 143' INT TERM
echo "=== OLMo seeds $SEEDS, REVERSE eval runner on GPUs $GPU_A,$GPU_B: ${#JOBS[@]} eval jobs, $WORKERS worker(s) ($(date)) ==="
echo "(olmo_seeds_results.tgz is refreshed after every finished job; packer log: pack.log)"
for ((w = 0; w < WORKERS; w++)); do worker $w & done
wait

echo "=== summary (all evals, both runners) ==="
for job in "${JOBS[@]}"; do job_done "$job" && echo "  ok      $job$(job_note "$job")" || echo "  MISSING $job"; done
package_results final
echo "REVERSE RUNNER DONE - fetch olmo_seeds_results.tgz ($(du -h olmo_seeds_results.tgz | cut -f1)); report: analysis/report.md"
[ -f analysis/report.md ] && head -n 80 analysis/report.md
