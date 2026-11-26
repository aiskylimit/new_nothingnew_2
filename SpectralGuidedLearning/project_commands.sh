#!/usr/bin/env bash
# Spectral experiment driver -- TRAIN everything, then EVAL, then compare.
# The IWC arm has its own driver (project_commands2.sh) so it can run on another GPU while
# this one is busy; the two touch disjoint data/, checkpoints/, logs/ and results/<tag>/ paths.
# Models: qwen25-7b, qwen3-8b (P-ALIGN's two student models). Dataset: s1K-1.1.
# Comment out any line you don't want to run.
#
# The r1-qwen-1.5b / r1-qwen-7b scripts are still in scripts/ but are out of the driver:
# r1-qwen-1.5b now trains on s1K-1.1 too (its own driver: project_commands_r1-qwen-1.5b.sh),
# r1-qwen-7b still trains on LIMO; neither is part of the P-ALIGN comparison.
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"
export PYTHONPATH="${BASE}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Which GPU(s) every phase script runs on (space-separated ids). Override: GPUS="0 1" ./project_commands.sh
CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-1}"

# ============================ TRAIN ============================
# per model: data -> capture (spectral + entropy) -> masks -> four matched SFT arms.
# IWC and IWC-Stable share exactly the spectral-selected token set; only weights differ.
# source /mnt/local/uvenvs/spectral_guided_learning_train/bin/activate
# qwen25-7b/qwen3-8b long-CoT arms đã train + eval xong -- tạm tắt; lượt này chỉ chạy arm answer-only.
# bash scripts/data/data_qwen25-7b.sh
# bash scripts/capture/capture_qwen25-7b.sh
# bash scripts/masks/masks_qwen25-7b.sh
# bash scripts/masks/iwc_qwen25-7b.sh
# bash scripts/spectral/spectral_qwen25-7b.sh
# bash scripts/sft/sft_qwen25-7b.sh
# bash scripts/iwc/train_iwc.sh qwen25-7b iwc
# bash scripts/iwc/train_iwc.sh qwen25-7b iwc-stable
# bash scripts/spectral/spectral_unsloth_qwen25-7b.sh
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen25-7b iwc-stable
# bash scripts/sft/sft_unsloth_qwen25-7b.sh
# answer-only SFT arm (ground-truth `solution`, no long CoT): data+masks in one step, no
# capture/spectral needed. Data prep and training (train_sft.py + DeepSpeed) both run in the
# main env (spectral_guided_learning.txt); the unsloth variant is kept commented for reference.
# bash scripts/data/data_answer.sh qwen25-7b
# bash scripts/sft/sft_answer.sh qwen25-7b
# bash scripts/sft/sft_answer_unsloth.sh qwen25-7b

# bash scripts/data/data_qwen3-8b.sh
# bash scripts/capture/capture_qwen3-8b.sh
# bash scripts/masks/masks_qwen3-8b.sh
# bash scripts/masks/iwc_qwen3-8b.sh
# bash scripts/spectral/spectral_qwen3-8b.sh
# bash scripts/sft/sft_qwen3-8b.sh
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc
# bash scripts/iwc/train_iwc.sh qwen3-8b iwc-stable
# bash scripts/spectral/spectral_unsloth_qwen3-8b.sh
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc
# bash scripts/iwc/train_iwc_unsloth.sh qwen3-8b iwc-stable
# bash scripts/sft/sft_unsloth_qwen3-8b.sh
# bash scripts/data/data_answer.sh qwen3-8b
# bash scripts/sft/sft_answer.sh qwen3-8b
# bash scripts/sft/sft_answer_unsloth.sh qwen3-8b

# ============================ EVAL =============================
# Start from a clean shell env so each eval script activates the vLLM env (spectral-guided-learning).
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
# only the answer-only checkpoints this run (the other arms are already in results/)

# bash scripts/eval/eval_qwen25-7b.sh
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/vanilla-qwen25-7b vanilla-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-qwen25-7b iwc-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-qwen25-7b iwc-stable-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-unsloth-qwen25-7b iwc-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/iwc-stable-unsloth-qwen25-7b iwc-stable-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/vanilla-unsloth-qwen25-7b vanilla-unsloth-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/answer-qwen25-7b answer-qwen25-7b
# bash scripts/eval/eval_qwen25-7b.sh checkpoints/answer-unsloth-qwen25-7b answer-unsloth-qwen25-7b

# bash scripts/eval/eval_qwen3-8b.sh
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-qwen3-8b vanilla-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-qwen3-8b iwc-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/iwc-stable-qwen3-8b iwc-stable-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/vanilla-unsloth-qwen3-8b vanilla-unsloth-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/answer-qwen3-8b answer-qwen3-8b
# bash scripts/eval/eval_qwen3-8b.sh checkpoints/answer-unsloth-qwen3-8b answer-unsloth-qwen3-8b

# ================= LoRA P-ALIGN answer-gain (current run) =================
# Single entry point: qwen25-7b on GPU 4 and qwen3-8b on GPU 5, concurrently. Each run does
# data -> answer-gain signal -> IWC-Stable weights (lambda=1) -> LoRA train (train seed 3407)
# -> eval on sampling seeds 42/43/44. Phases are resumable; override with e.g. GPU_7B=0 GPU_8B=1.
# Both arms are named ...-l1-lora-s3407-... so they never overwrite the earlier seed-42 / lambda runs.
export PROJECT_ENV="${PROJECT_ENV:-/mnt/local/uvenvs/spectral_guided_learning}"
export SEED=3407 ARMS=gain
# Method-only tuning of the 7B (8B is already fine -> RUN_8B=0). CONFIGS = "tau:clip:lambda" entries.
# lr / batch (7B: 8) / seed / LoRA recipe stay identical to the NLL baseline. (2:2) was the earlier default.
# 7B at batch 8 takes 4x more optimizer steps than before, so sharp weights add gradient variance:
# try milder weights (lower lambda) and tau/clip between the old (2,2) and the sharper (1,3).
# Override: CONFIGS="1.0:3.0:0.3 2.0:2.0:0.5" RUN_8B=1 ./project_commands.sh
export RUN_8B="${RUN_8B:-0}"
export EVAL_SEEDS="${EVAL_SEEDS:-42}"
export CONFIGS="${CONFIGS:-2.0:2.0:0.5 2.0:2.0:0.3 1.0:3.0:0.3 1.0:3.0:0.5 1.0:2.0:1.0}"
GPU_7B="${GPU_7B:-4}"
GPU_8B="${GPU_8B:-5}"
mkdir -p logs

# ---- preflight: print the environment and fail loudly BEFORE launching anything ----
echo "[preflight] host=$(hostname) user=$(whoami) pwd=$(pwd)"
echo "[preflight] PROJECT_ENV=${PROJECT_ENV}"
echo "[preflight] GPU_7B=${GPU_7B} GPU_8B=${GPU_8B} SEED=${SEED} ARMS=${ARMS}"
if [[ ! -f "${PROJECT_ENV}/bin/activate" ]]; then
  echo "[preflight] ERROR: ${PROJECT_ENV}/bin/activate not found." >&2
  echo "[preflight] venv-like dirs found nearby (re-run with PROJECT_ENV=<one of these>):" >&2
  for d in /mnt/local/uvenvs /mnt/*/uvenvs "${HOME}/uvenvs" "${HOME}/.venvs" "$(dirname "${PROJECT_ENV}")"; do
    [[ -d "$d" ]] && { echo "  in $d:" >&2; ls -1 "$d" 2>&1 | sed 's/^/    /' >&2; }
  done
  exit 2
fi
if ! "${PROJECT_ENV}/bin/python" -c "import torch, vllm" 2>logs/preflight-import.log; then
  echo "[preflight] WARNING: import torch/vllm failed in ${PROJECT_ENV}:" >&2
  tail -n 5 logs/preflight-import.log >&2
fi
echo "[preflight] python=$("${PROJECT_ENV}/bin/python" --version 2>&1)"
echo "[preflight] nvidia-smi (memory used on chosen GPUs):"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /' || true

show_tail() {   # $1 = label, $2 = log file
  echo "------ $1: last 30 lines of $2 ------" >&2
  tail -n 30 "$2" >&2
  echo "------ end $1 ------" >&2
}

RC=0
for CFG in ${CONFIGS}; do
  IFS=: read -r IWC_TEMPERATURE IWC_CLIP IWC_INTERPOLATION <<< "${CFG}"
  export IWC_TEMPERATURE IWC_CLIP IWC_INTERPOLATION
  CTAG="t${IWC_TEMPERATURE}-c${IWC_CLIP}-l${IWC_INTERPOLATION}"
  echo "[sweep] ===== tau=${IWC_TEMPERATURE} clip=${IWC_CLIP} lambda=${IWC_INTERPOLATION} ====="
  GPUS="${GPU_7B}" bash project_commands_lora_palign_qwen25-7b.sh > >(tee "logs/run-qwen25-7b-s3407-${CTAG}.log" | sed -u "s/^/[7b ${CTAG}] /") 2>&1 &
  PID_7B=$!
  echo "[launch] qwen25-7b pid=${PID_7B} gpu=${GPU_7B} log=logs/run-qwen25-7b-s3407-${CTAG}.log"
  PID_8B=""
  if [[ "${RUN_8B}" == 1 ]]; then
    # stagger the start so the two 16GB model loads don't hit host RAM at the same moment
    # (also catch an immediate crash, e.g. bad env, instead of silently sleeping)
    for _ in $(seq 1 12); do
      sleep 10
      kill -0 "${PID_7B}" 2>/dev/null || break
    done
    GPUS="${GPU_8B}" bash project_commands_lora_palign_qwen3-8b.sh > >(tee "logs/run-qwen3-8b-s3407-${CTAG}.log" | sed -u "s/^/[8b ${CTAG}] /") 2>&1 &
    PID_8B=$!
    echo "[launch] qwen3-8b pid=${PID_8B} gpu=${GPU_8B} log=logs/run-qwen3-8b-s3407-${CTAG}.log"
  fi
  wait "${PID_7B}" || { E=$?; RC=1; echo "qwen25-7b ${CTAG} FAILED (exit ${E})" >&2; show_tail qwen25-7b "logs/run-qwen25-7b-s3407-${CTAG}.log"; }
  if [[ -n "${PID_8B}" ]]; then
    wait "${PID_8B}" || { E=$?; RC=1; echo "qwen3-8b ${CTAG} FAILED (exit ${E})" >&2; show_tail qwen3-8b "logs/run-qwen3-8b-s3407-${CTAG}.log"; }
  fi
done
echo "[done] sweep finished, RC=${RC}. Full logs: logs/run-*-s3407-*.log"

# =========================== PRINT RESULTS ==========================
# Print every result of the two runs: per sampling seed (42/43/44) and the mean over seeds.
"${PROJECT_ENV}/bin/python" - <<'PY' || true
import json, os
bench = ["aime24", "aime25", "amc12", "math500"]
eval_seeds = [int(s) for s in os.environ.get("EVAL_SEEDS", "42").split()]
for cfg in os.environ["CONFIGS"].split():
  tau, clip, lam = (float(x) for x in cfg.split(":"))
  suffix = "" if (tau == 2 and clip == 2) else f"-t{tau:g}-c{clip:g}"
  ltag = "l" + (f"{lam:g}" if lam >= 1 else f"{lam:g}".replace("0.", "0"))
  models = ("qwen25-7b", "qwen3-8b") if os.environ.get("RUN_8B") == "1" else ("qwen25-7b",)
  for model in models:
    tag = f"iwc-gain-{ltag}{suffix}-lora-s3407-{model}-palign"
    runs = {seed: (f"results/{tag}" if seed == 42 else f"results_evalseed/{tag}-e{seed}") for seed in eval_seeds}
    for metric in ("pass@1", "pass@3"):
        print(f"\n=== {tag} | {metric} ===")
        print(f"{'eval seed':<10}" + "".join(f"{b:>10}" for b in bench) + f"{'Avg':>10}")
        rows = []
        for seed, d in runs.items():
            f = os.path.join(d, "summary.json")
            if not os.path.exists(f):
                print(f"{seed:<10}(missing: {f})"); continue
            r = {x["benchmark"]: 100 * x[metric] for x in json.load(open(f))}
            vals = [r.get(b, float("nan")) for b in bench]
            rows.append(vals)
            print(f"{seed:<10}" + "".join(f"{v:>9.2f}%" for v in vals) + f"{sum(vals)/len(vals):>9.2f}%")
        if len(rows) > 1:
            m = [sum(c) / len(c) for c in zip(*rows)]
            print(f"{'mean':<10}" + "".join(f"{v:>9.2f}%" for v in m) + f"{sum(m)/len(m):>9.2f}%")
PY

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
"${PROJECT_ENV}/bin/python" -m sgl.eval.compare
echo; cat results/comparison-table.md || true
exit "${RC}"
