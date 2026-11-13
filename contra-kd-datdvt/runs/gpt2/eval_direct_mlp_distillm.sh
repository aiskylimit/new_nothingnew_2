#!/usr/bin/env bash
# Evaluate the existing direct-MLP DistiLLM student without training it again.
set -euo pipefail

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
CKPT="direct_mlp_t0/distillm/0.1B_1.5B"
MODEL_PATH="${BASE_PATH}/results/gpt2/train/${CKPT}"
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Missing trained DistiLLM checkpoint: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES=6
export PET_STANDALONE=1
EVAL_PORT=${EVAL_PORT:-$(( ${BASE_PORT:-20120} + 20 ))}

for benchmark in dolly self_inst vicuna sinst; do
    for seed in 10 20 30 40 50; do
        bash "${BASE_PATH}/scripts/gpt2/eval/eval_main_${benchmark}.sh" \
            "${BASE_PATH}" "${EVAL_PORT}" 1 "${CKPT}" \
            --seed "${seed}" --eval-batch-size 64
    done
done
