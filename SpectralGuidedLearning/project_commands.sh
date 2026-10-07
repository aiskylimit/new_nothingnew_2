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
export SEED=3407 IWC_INTERPOLATION=1.0 ARMS=gain
GPU_7B="${GPU_7B:-4}"
GPU_8B="${GPU_8B:-5}"
mkdir -p logs
GPUS="${GPU_7B}" bash project_commands_lora_palign_qwen25-7b.sh > logs/run-qwen25-7b-s3407.log 2>&1 &
PID_7B=$!
# stagger the start so the two 16GB model loads don't hit host RAM at the same moment
sleep 120
GPUS="${GPU_8B}" bash project_commands_lora_palign_qwen3-8b.sh > logs/run-qwen3-8b-s3407.log 2>&1 &
PID_8B=$!
RC=0
wait "${PID_7B}" || { echo "qwen25-7b run FAILED (see logs/run-qwen25-7b-s3407.log)" >&2; RC=1; }
wait "${PID_8B}" || { echo "qwen3-8b run FAILED (see logs/run-qwen3-8b-s3407.log)" >&2; RC=1; }

# =========================== PRINT RESULTS ==========================
# Print every result of the two runs: per sampling seed (42/43/44) and the mean over seeds.
"${PROJECT_ENV}/bin/python" - <<'PY' || true
import json, os
bench = ["aime24", "aime25", "amc12", "math500"]
for model in ("qwen25-7b", "qwen3-8b"):
    tag = f"iwc-gain-l1-lora-s3407-{model}-palign"
    runs = {42: f"results/{tag}", 43: f"results_evalseed/{tag}-e43", 44: f"results_evalseed/{tag}-e44"}
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
        if rows:
            m = [sum(c) / len(c) for c in zip(*rows)]
            print(f"{'mean':<10}" + "".join(f"{v:>9.2f}%" for v in m) + f"{sum(m)/len(m):>9.2f}%")
PY

# =========================== COMPARE ==========================
# writes results/comparison-table.md and results/eval-summary.json
"${PROJECT_ENV}/bin/python" -m sgl.eval.compare
echo; cat results/comparison-table.md || true
exit "${RC}"
