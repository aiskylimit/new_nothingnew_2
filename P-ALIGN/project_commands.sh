#!/bin/bash
# All run commands for P-ALIGN on Qwen2.5-7B-Instruct (full-parameter SFT).
# No Hub upload, no git push. Train JSON is tracked as data/palign_sft_qwen2.5-7b.json.gz (unpacked by cmd_data)
#
# Usage:
#   bash project_commands.sh            # data check + train + eval
#   bash project_commands.sh env
#   bash project_commands.sh data
#   bash project_commands.sh train
#   bash project_commands.sh eval       # needs full SFT checkpoint (or MODEL=...)
#
# Eval base checkpoint only (skip train):
#   MODEL=/mnt/local/aiskylimit_new_nothing/P-ALIGN/models/Qwen2.5-7B-Instruct bash project_commands.sh eval
#
# Tiny 0.5B flow check (does not replace this 7B script):
#   bash project_commands_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ASSET_ROOT="${ASSET_ROOT:-/mnt/local/aiskylimit_new_nothing/P-ALIGN}"
MODEL_PATH="${MODEL_PATH:-$ASSET_ROOT/models/Qwen2.5-7B-Instruct}"
TRAIN_OUT="${TRAIN_OUT:-output/palign-qwen2.5-7b-full}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
export PALIGN_ASSET_ROOT="$ASSET_ROOT"
export PALIGN_DATA_DIR="$DATA_DIR"
if [ -f "${PALIGN_VENV:-/mnt/local/uvenvs/p-align}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${PALIGN_VENV:-/mnt/local/uvenvs/p-align}/bin/activate"
elif [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
else
  echo "no python env (PALIGN_VENV or /mnt/local/uvenvs/p-align or /venv/main)" >&2
  exit 1
fi

export WANDB_DISABLED=true
export WANDB_MODE=disabled
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Make CUDA_VISIBLE_DEVICES indices match nvidia-smi (PCI order) instead of FASTEST_FIRST.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29330}"
EFFECTIVE_BATCH=32
PER_DEVICE_BS=1
STAGE="${1:-all}"

pick_ckpt() {
  TRAIN_OUT="$TRAIN_OUT" python - <<'PY'
import glob, json, os
root = os.environ["TRAIN_OUT"]
best = None
for d in glob.glob(os.path.join(root, "checkpoint-*")):
    st = os.path.join(d, "trainer_state.json")
    if not os.path.exists(st):
        continue
    ep = json.load(open(st))["epoch"]
    if best is None or abs(ep - 3.0) < abs(best[0] - 3.0):
        best = (ep, d)
if best is not None:
    print(best[1])
elif os.path.isfile(os.path.join(root, "config.json")):
    print(root)
else:
    raise SystemExit(f"no checkpoint found under {root}")
PY
}

cmd_env() {
  python -c "import torch, transformers, llamafactory, vllm; print('env ok')"
}

cmd_data() {
  mkdir -p "$DATA_DIR/raw" output/log output/result
  if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ] && [ -f "$DATA_DIR/palign_sft_qwen2.5-7b.json.gz" ]; then
    gunzip -kc "$DATA_DIR/palign_sft_qwen2.5-7b.json.gz" > "$DATA_DIR/palign_sft_qwen2.5-7b.json"
  fi
  if [ ! -f "$DATA_DIR/palign_sft_qwen2.5-7b.json" ]; then
    echo "missing local train file $DATA_DIR/palign_sft_qwen2.5-7b.json" >&2
    exit 1
  fi
  for f in "$DATA_DIR/raw/aime25.jsonl" "$DATA_DIR/raw/aime24.jsonl" "$DATA_DIR/raw/amc12.jsonl" "$DATA_DIR/raw/math500.jsonl"; do
    if [ ! -f "$f" ]; then
      python src/fetch_eval.py
      break
    fi
  done
}

cmd_train() {
  mkdir -p output/log
  if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "missing model checkpoint: $MODEL_PATH" >&2
    exit 1
  fi
  WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
  GRAD_ACCUM=$((EFFECTIVE_BATCH / (PER_DEVICE_BS * WORLD_SIZE)))
  echo "SFT nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH} model=${MODEL_PATH}"
  # Cluster pods export PET_RDZV_* (c10d rendezvous on a pod hostname) which
  # overrides --master_addr and hangs at "Rendezvous'ing worker group" on a
  # single node; --standalone forces a local rendezvous.
  if [ "$NNODES" -eq 1 ]; then
    LAUNCH_ARGS=(--standalone --nproc_per_node "$NPROC_PER_NODE")
  else
    LAUNCH_ARGS=(--nproc_per_node "$NPROC_PER_NODE" --nnodes "$NNODES" --node_rank "$RANK"
                 --rdzv_backend static --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT")
  fi
  torchrun "${LAUNCH_ARGS[@]}" \
    src/train.py configs/qwen25_7b_palign_sft.yaml \
    model_name_or_path="$MODEL_PATH" \
    output_dir="$TRAIN_OUT" \
    gradient_accumulation_steps="$GRAD_ACCUM"
  # num_train_epochs=3 with save_strategy=epoch; eval the checkpoint closest to epoch 3.
  BENCH_CKPT="$(pick_ckpt)"
  echo "benchmark checkpoint: $BENCH_CKPT"
}

cmd_eval() {
  mkdir -p output/result
  MODEL="${MODEL:-$(pick_ckpt)}"
  echo "evaluating model=$MODEL"
  python src/test.py \
    --model "$MODEL" \
    --input_files "$DATA_DIR/raw/aime25.jsonl" "$DATA_DIR/raw/aime24.jsonl" "$DATA_DIR/raw/amc12.jsonl" "$DATA_DIR/raw/math500.jsonl" \
    --output_files output/result/aime25.jsonl output/result/aime24.jsonl output/result/amc12.jsonl output/result/math500.jsonl \
    --batch_size 1000 \
    --n 3 \
    --temperature 0.6 \
    --top_p 0.9 \
    --repetition_penalty 1.05 \
    --max_tokens 4096
  for f in output/result/aime25.jsonl output/result/aime24.jsonl output/result/amc12.jsonl output/result/math500.jsonl; do
    python src/evaluation.py --input_path "$f" --output_path "${f%.jsonl}_scored.jsonl"
  done
  python src/report.py --out output/eval_results.txt
  echo "wrote output/eval_results.txt"
}

case "$STAGE" in
  env) cmd_env ;;
  data) cmd_data ;;
  train) cmd_data; cmd_train ;;
  eval) cmd_data; cmd_eval ;;
  all) cmd_env; cmd_data; cmd_train; cmd_eval ;;
  *)
    echo "usage: bash project_commands.sh [all|env|data|train|eval]" >&2
    exit 1
    ;;
esac
