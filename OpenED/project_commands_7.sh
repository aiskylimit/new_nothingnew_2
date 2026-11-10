#!/usr/bin/env bash
# GENEVA baselines on another model family, 5 perms. This file runs Llama-3.2-1B-Instruct;
# project_commands_8.sh runs the same jobs for Gemma-3-1b-it (MODEL=gemma). Ours comes later,
# in its own script.
#
#   bash project_commands_7.sh                              # Llama, gpus 2 and 3
#   DRY=1 bash project_commands_7.sh                        # print the plan, train nothing
#   PERMS="0" GPUS="2" bash project_commands_7.sh
#
# Jobs per perm, in the order they are handed out: shared task0, IncLoRA, MIGU, DistiLLM, AMiD
# (25 jobs). A free GPU takes the first job that can start: DistiLLM and AMiD start from the
# shared task0 of their perm, so until it is trained the GPU moves on to the next perm's jobs.
#
# Same protocol as the Qwen GENEVA baselines: geneva_b10_perm data, LoRA r16/a64, 5 epochs,
# lr 2e-4, effective batch 32, greedy eval, untuned hyperparameters. Only the micro-batch is
# 8 x 4 (Qwen used 32 x 1 and 16 x 2): a 1B model, and Gemma's 262k vocabulary, need more
# memory per row.
# Run names fold the model into the protocol tag: dist_amid_perm0_geneva_llama1b_v2_s42,
# cllora_migu_perm0_geneva_gemma1b_v2_s42, dist_shared_task0_perm0_geneva_llama1b_v2_s42.
#
# Knobs: MODEL [llama] (llama | gemma), GPUS [2 3], SLOTS jobs per GPU [1], NEED_MB free MiB a
# job waits for [80000], PERMS [0 1 2 3 4], SEED [42], VENV / PY / ENV_BIN as in project_commands.sh.
# Llama and Gemma keep separate claims, pool logs and ports, so 7 and 8 can run at the same time.
# Safe to re-run: finished runs are skipped, a run another process is training is left alone,
# and a crashed partial one is moved to results/qwen3/ced/_failed/ (never deleted) and retrained.
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
MODEL=${MODEL:-llama}
GPUS=(${GPUS:-2 3})
SLOTS=${SLOTS:-1}
NEED_MB=${NEED_MB:-80000}
PERMS=${PERMS:-"0 1 2 3 4"}
SEED=${SEED:-42}
MB=8   # micro-batch; x 4 accumulation = 32
# local dir, official repo, ungated copy of the same weights, run-name tag, first master port
case ${MODEL} in
    llama) MPATH=models/Llama-3.2-1B-Instruct; REPO=meta-llama/Llama-3.2-1B-Instruct
           MIRROR=unsloth/Llama-3.2-1B-Instruct; TAG=llama1b; PORT0=29700 ;;
    gemma) MPATH=models/gemma-3-1b-it; REPO=google/gemma-3-1b-it
           MIRROR=unsloth/gemma-3-1b-it; TAG=gemma1b; PORT0=29800 ;;
    *) echo "unknown MODEL '${MODEL}' (llama|gemma)"; exit 1 ;;
esac
PRE=geneva_b10_perm
PROTO=geneva_${TAG}_v2
R=results/qwen3/ced
CLAIMS=logs/geneva_${MODEL}_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/geneva_${MODEL}_pool.log
log () { echo "[${MODEL} $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# ---------------------------------------------------------------- environment (as project_commands.sh)
if [ -z "${VENV:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f /mnt/local/uvenvs/opened/bin/activate ]; then
    VENV=/mnt/local/uvenvs/opened
fi
if [ -n "${VENV:-}" ]; then set +u; source "${VENV}/bin/activate"; set -u; fi
PY=${PY:-$(command -v python || command -v python3)}
ENV_BIN=${ENV_BIN:-$(dirname "${PY}")}
export PY ENV_BIN
for v in $(compgen -e | grep '^PET_' || true); do unset "${v}"; done
if [ -f .env ] && [ -z "${HF_TOKEN:-}" ]; then set -a; . ./.env; set +a; fi   # gated Llama/Gemma repos

log "=== GENEVA baselines, ${MODEL}, perms '${PERMS}', gpus ${GPUS[*]} x ${SLOTS} (DRY=${DRY}) ==="
# weights: download.txt puts them in models/; otherwise the official repo, then the ungated copy
if [ ! -f "${MPATH}/config.json" ]; then
    log "  ${MPATH} missing, downloading ${REPO}"
    if [ "${DRY}" != "1" ]; then
        for repo in "${REPO}" "${MIRROR}"; do
            HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${PY}" -c "import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['*.json', '*.safetensors', '*.model'])" \
                "${repo}" "${MPATH}" >> "${LOG}" 2>&1 && [ -f "${MPATH}/config.json" ] \
                && { echo "${repo}" > "${MPATH}/SOURCE"; log "  got ${repo}"; break; }
            log "  ${repo} failed (gated repo without access?), see ${LOG}"
        done
        [ -f "${MPATH}/config.json" ] || { log "no weights for ${MODEL}"; exit 1; }
    fi
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # local weights only from here on
export MODEL_PATH=${MPATH} MODEL_TYPE=${MODEL} MODEL_TAG=${TAG} MB_CAP=${MB}   # read by the runners

# GENEVA ships as geneva_all.tar.gz (download.txt); unpack it as project_commands.sh does. A lock,
# since 7 and 8 may start together.
if [ -e geneva_all.tar.gz ] && mkdir logs/geneva_unpack.lock 2>/dev/null; then
    log "  unpacking geneva_all.tar.gz"
    [ "${DRY}" = "1" ] || { "${PY}" -c "import sys, tarfile; tarfile.open(sys.argv[1]).extractall('.', filter='data')" geneva_all.tar.gz && rm -f geneva_all.tar.gz; }
    rmdir logs/geneva_unpack.lock
fi
while [ -d logs/geneva_unpack.lock ] && [ -e geneva_all.tar.gz ]; do sleep 10; done   # the other one unpacks
for p in ${PERMS}; do
    [ -s "data/${PRE}${p}/streams.json" ] || { log "missing data/${PRE}${p}, get geneva_all.tar.gz (download.txt)"; exit 1; }
done

# ---------------------------------------------------------------- jobs
run_of () {  # $1 = kind $2 = perm -> run name
    case $1 in
        t0)            echo "dist_shared_task0_perm$2_${PROTO}_s${SEED}" ;;
        inclora|migu)  echo "cllora_$1_perm$2_${PROTO}_s${SEED}" ;;
        distillm|amid) echo "dist_$1_perm$2_${PROTO}_s${SEED}" ;;
    esac
}
live () { pgrep -f -- "$1" > /dev/null; }   # some process has the run name on its command line
park () {
    local dst="${R}/_failed/$1_$(date +%Y%m%d_%H%M)"
    log "  moving partial ${R}/$1 -> ${dst}"
    [ "${DRY}" = "1" ] || mv "${R}/$1" "${dst}"
}
wait_gpu () {  # $1 = gpu: block until it has NEED_MB free (a snapshot, so slots also stagger)
    local free
    [ "${DRY}" = "1" ] && return 0
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -dc '0-9')
        [ "${free:-0}" -ge "${NEED_MB}" ] && return 0
        log "  gpu$1 has ${free:-?} MiB free < ${NEED_MB}, waiting"; sleep 120
    done
}
ensure_tokenized () {  # $1 = perm: the base task data in this model's tokenizer
    local p=$1 n t out
    n=$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "data/${PRE}${p}/streams.json")
    for t in $(seq 0 $((n - 1))); do
        out="processed_data/${PRE}${p}/${t}"
        [ -f "${out}/${MODEL}/train_0.idx" ] && continue
        log "  tokenising ${PRE}${p} task${t}"
        [ "${DRY}" = "1" ] && continue
        PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${PRE}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${MPATH}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type "${MODEL}" > "logs/tok_${PRE}${p}_t${t}_${MODEL}.log" 2>&1 || return 1
    done
}

run_job () {  # $1 = gpu $2 = port $3 = kind $4 = perm -> 0 done, 1 failed, 2 busy elsewhere
    local g=$1 port=$2 k=$3 p=$4 rc=0 run t0 jlog
    run=$(run_of "${k}" "${p}"); t0=$(run_of t0 "${p}"); jlog="logs/models7_${run}.log"
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    live "${run}" && return 2
    case ${k} in
        distillm|amid)
            [ -f "${R}/${t0}/.complete" ] || [ "${DRY}" = "1" ] \
                || { log "  FAILED ${run}: no shared task0 ${t0}"; return 1; } ;;
    esac
    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    case ${k} in
        t0)
            ensure_tokenized "${p}" || { log "  FAILED ${run}: tokenising, see logs/tok_${PRE}${p}_t*_${MODEL}.log"; return 1; }
            PERM="${p}" GPU="${g}" PROTOCOL="${PROTO}" SEED="${SEED}" DATA_PREFIX="${PRE}" DIST_METHODS="" \
            MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh > "${jlog}" 2>&1 || rc=$? ;;
        inclora|migu)
            bash scripts/qwen/ced/run_cllora.sh --method "${k}" --data-root "data/${PRE}${p}" \
                --protocol "${PROTO}" --seed "${SEED}" --batch-size "${MB}" --grad-accum $((32 / MB)) \
                --eval-batch-size 16 --gpu "${g}" --py "${PY}" > "${jlog}" 2>&1 || rc=$? ;;
        distillm|amid)
            PERM="${p}" GPU="${g}" PROTOCOL="${PROTO}" SEED="${SEED}" DATA_PREFIX="${PRE}" DIST_METHODS="${k}" \
            MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh > "${jlog}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog}"
    return 1
}

JOBS=()   # kind:perm, in hand-out order
for p in ${PERMS}; do
    for k in t0 inclora migu distillm amid; do JOBS+=("${k}:${p}"); done
done
ready () {  # $1 = kind $2 = perm: 0 when the job may start now
    case $1 in t0|inclora|migu) return 0 ;; esac
    [ -e "${CLAIMS}/t0_$2.done" ]   # the shared task0 has been dealt with (done or failed)
}
worker () {  # $1 = gpu $2 = slot: take the first unclaimed ready job, then start over from the top
    local g=$1 s=$2 port=$((PORT0 + 10 * $1 + $2)) j k p left rc n=0
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))   # later slots see the earlier job's memory
    while :; do
        left=0
        for j in "${JOBS[@]}"; do
            IFS=: read -r k p <<< "${j}"
            [ -e "${CLAIMS}/${k}_${p}.done" ] && continue
            left=1
            ready "${k}" "${p}" || continue
            mkdir "${CLAIMS}/${k}_${p}" 2>/dev/null || continue   # atomic claim
            rc=0; run_job "${g}" "${port}" "${k}" "${p}" || rc=$?
            if [ "${rc}" -eq 2 ]; then   # trained by another process: look again later
                rmdir "${CLAIMS}/${k}_${p}"; continue
            fi
            [ "${rc}" -eq 0 ] || n=$((n + 1))
            touch "${CLAIMS}/${k}_${p}.done"
            continue 2
        done
        [ "${left}" = "0" ] && break
        if [ "${DRY}" = "1" ]; then sleep 1; else sleep 60; fi
    done
    log "worker gpu${g}/slot${s} finished, ${n} failed"
}

log "  ${#JOBS[@]} jobs; pool log ${LOG}, per run logs/models7_<run>.log"
pids=()
for s in $(seq 1 "${SLOTS}"); do
    for g in "${GPUS[@]}"; do worker "${g}" "${s}" & pids+=($!); done
done
wait "${pids[@]}"
log "=== all done: grep FAILED ${LOG}; results in ${R}/*_${PROTO}_s${SEED}/ ==="
[ "${DRY}" = "1" ] || bash gather_logs.sh "geneva_${MODEL}_$(date +%Y%m%d_%H%M)" || log "gather_logs.sh failed, run it by hand"
