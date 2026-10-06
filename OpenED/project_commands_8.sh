#!/usr/bin/env bash
# GENEVA baselines on Gemma-3-1b-it: project_commands_7.sh with MODEL=gemma (same 25 jobs per
# model: shared task0, IncLoRA, MIGU, DistiLLM, AMiD over 5 perms). See that file for details.
#
#   GPUS="..." bash project_commands_8.sh
#   DRY=1 GPUS="0" bash project_commands_8.sh               # print the plan, train nothing
#
# GPUS has no default here: 7 already takes 2 and 3, and a Gemma job next to a Llama job on the
# same GPU risks running both out of memory.
set -uo pipefail
GPUS=${GPUS:?set GPUS, the GPUs for Gemma, e.g. GPUS=\"4 5\"} MODEL=gemma \
    exec bash "$(dirname "$0")/project_commands_7.sh"
