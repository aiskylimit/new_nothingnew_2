#!/usr/bin/env bash
# Other model families on GENEVA: Llama-3.2-1B-Instruct and Gemma-3-1b-it, 5 perms each.
#
#   bash project_commands_7.sh                              # gpus 2 and 3: baselines, then Ours
#   DRY=1 bash project_commands_7.sh                        # print the plan, train nothing
#   PHASES="base" MODELS="llama" PERMS="0" GPUS="2" bash project_commands_7.sh
#
# Jobs, in the order they are handed out (a free GPU takes the first job whose inputs exist):
#   base  per perm and model: shared task0, IncLoRA, MIGU, DistiLLM, AMiD
#   ours  per perm and model, Qwen3-0.6B included (its GENEVA baselines already exist):
#         shared task0 (reused when present), f12_pl, f12_pl + SD
#         none of these starts before every baseline job has finished (done or failed)
# f12_pl  = ours_queue.sh OURS_VARIANT=h2: pseudo-labels with dedup + lexicon, no confidence
#           filter, replay x5, SFKL token KD + span loss (2.0, cosine) on replay rows, kd 0.9,
#           micro-batch 2 x 16 as on ACE (the CE/KD mix is decided per micro-batch, so a
#           larger micro-batch would change the recipe), from the shared task0
# + SD    = the same plus on-policy self-distillation with the SDFT defaults of run_ced_v2.sh
#           (w 1.0, mu 0.99, T 1.0, top-p 1.0, forward KL): ACE g1_full minus its confidence filter
#
# Same protocol as the Qwen GENEVA baselines: geneva_b10_perm data, LoRA r16/a64, 5 epochs,
# lr 2e-4, effective batch 32, greedy eval, untuned hyperparameters. Per model only:
#   * micro-batch 8 x 4 for the baselines and task0 (Qwen used 32 x 1 and 16 x 2; the 1B models
#     and Gemma's 262k vocabulary need more memory per row)
#   * span-loss layers at the same relative depth as Qwen's 22 25 28 of 28: Llama 13 14 16 of 16,
#     Gemma 20 23 26 of 26
# Run names fold the model into the protocol tag: dist_amid_perm0_geneva_llama1b_v2_s42,
# cllora_migu_perm0_geneva_gemma1b_v2_s42, ours_h2_sd_perm0_geneva_llama1b_v2_s42. Qwen keeps
# geneva_v2, so its task0 is reused when results/qwen3/ced/dist_shared_task0_perm<p>_geneva_v2_s42
# is copied over from the host that trained the Qwen baselines.
#
# Knobs: GPUS [2 3], SLOTS jobs per GPU [1], NEED_MB free MiB a job waits for
# [80000], MODELS [llama gemma], OURS_MODELS [qwen llama gemma], PHASES [base ours],
# PERMS [0 1 2 3 4], SEED [42], VENV / PY / ENV_BIN as in project_commands.sh.
# Safe to re-run: finished runs are skipped, a run another process is training is left alone,
# and a crashed partial one is moved to results/qwen3/ced/_failed/ (never deleted) and retrained.
set -uo pipefail
cd "$(dirname "$0")"

DRY=${DRY:-0}
GPUS=(${GPUS:-2 3})
SLOTS=${SLOTS:-1}
NEED_MB=${NEED_MB:-80000}
MODELS=${MODELS:-"llama gemma"}
OURS_MODELS=${OURS_MODELS:-"qwen ${MODELS}"}
PHASES=${PHASES:-"base ours"}
PERMS=${PERMS:-"0 1 2 3 4"}
SEED=${SEED:-42}
PRE=geneva_b10_perm
R=results/qwen3/ced
CLAIMS=logs/models7_claims
mkdir -p logs "${R}/_failed"
rm -rf "${CLAIMS}"; mkdir -p "${CLAIMS}"
LOG=logs/geneva_models_pool.log
log () { echo "[models7 $(date '+%F %T')] $*" | tee -a "${LOG}"; }

# local dir, official repo, ungated copy of the same weights, --model-type, run-name tag, layers
declare -A M_PATH=([qwen]=models/Qwen3-0.6B [llama]=models/Llama-3.2-1B-Instruct [gemma]=models/gemma-3-1b-it)
declare -A M_REPO=([qwen]=Qwen/Qwen3-0.6B [llama]=meta-llama/Llama-3.2-1B-Instruct [gemma]=google/gemma-3-1b-it)
declare -A M_MIRROR=([qwen]=Qwen/Qwen3-0.6B [llama]=unsloth/Llama-3.2-1B-Instruct [gemma]=unsloth/gemma-3-1b-it)
declare -A M_TAG=([qwen]="" [llama]=llama1b [gemma]=gemma1b)
declare -A M_LAYERS=([qwen]="22 25 28" [llama]="13 14 16" [gemma]="20 23 26")
declare -A M_MBCAP=([qwen]="" [llama]=8 [gemma]=8)   # baselines + task0; empty = Qwen's own

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

fetch_model () {  # $1 = model: download it into models/ unless it is there
    local m=$1 dir=${M_PATH[$1]} repo
    [ -f "${dir}/config.json" ] && return 0
    log "  ${dir} missing, downloading ${M_REPO[$m]}"
    [ "${DRY}" = "1" ] && return 0
    for repo in "${M_REPO[$m]}" "${M_MIRROR[$m]}"; do
        HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${PY}" -c "import sys; from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2], allow_patterns=['*.json', '*.safetensors', '*.model'])" \
            "${repo}" "${dir}" >> "${LOG}" 2>&1 && [ -f "${dir}/config.json" ] \
            && { echo "${repo}" > "${dir}/SOURCE"; log "  got ${repo}"; return 0; }
        log "  ${repo} failed (gated repo without access?), see ${LOG}"
    done
    return 1
}
log "=== GENEVA, models '${MODELS}', ours on '${OURS_MODELS}', phases '${PHASES}', perms '${PERMS}', gpus ${GPUS[*]} x ${SLOTS} (DRY=${DRY}) ==="
for m in $(echo ${MODELS} ${OURS_MODELS} | tr ' ' '\n' | sort -u); do
    [ -n "${M_PATH[$m]:-}" ] || { log "unknown model '${m}' (qwen|llama|gemma)"; exit 1; }
    fetch_model "${m}" || { log "no weights for ${m}"; exit 1; }
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # local weights only from here on

# GENEVA ships as geneva_all.tar.gz (download.txt); unpack it as project_commands.sh does
for tgz in geneva_all.tar.gz; do
    [ -e "${tgz}" ] || continue
    log "  unpacking ${tgz}"
    [ "${DRY}" = "1" ] || { "${PY}" -c "import sys, tarfile; tarfile.open(sys.argv[1]).extractall('.', filter='data')" "${tgz}" && rm -f "${tgz}"; }
done
for p in ${PERMS}; do
    [ -s "data/${PRE}${p}/streams.json" ] || { log "missing data/${PRE}${p}, get geneva_all.tar.gz (download.txt)"; exit 1; }
done

# ---------------------------------------------------------------- jobs
proto_of () { echo "geneva${M_TAG[$1]:+_${M_TAG[$1]}}_v2"; }
run_of () {  # $1 = kind $2 = model $3 = perm -> run name
    local pr; pr=$(proto_of "$2")
    case $1 in
        t0)            echo "dist_shared_task0_perm$3_${pr}_s${SEED}" ;;
        inclora|migu)  echo "cllora_$1_perm$3_${pr}_s${SEED}" ;;
        distillm|amid) echo "dist_$1_perm$3_${pr}_s${SEED}" ;;
        f12)           echo "ours_h2_perm$3_${pr}_s${SEED}" ;;
        f12sd)         echo "ours_h2_sd_perm$3_${pr}_s${SEED}" ;;
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
ensure_tokenized () {  # $1 = model $2 = perm: the base task data in that model's tokenizer
    local m=$1 p=$2 typ=${1} n t out
    n=$("${PY}" -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "data/${PRE}${p}/streams.json")
    for t in $(seq 0 $((n - 1))); do
        out="processed_data/${PRE}${p}/${t}"
        [ -f "${out}/${typ}/train_0.idx" ] && continue
        log "  tokenising ${PRE}${p} task${t} for ${m}"
        [ "${DRY}" = "1" ] && continue
        PYTHONPATH=. "${PY}" tools/process_data.py \
            --data-dir "data/${PRE}${p}/${t}/" --processed-data-dir "${out}" \
            --model-path "${M_PATH[$m]}" --data-process-workers 4 \
            --max-prompt-length 460 --t-max-prompt-length 640 \
            --dev-num 1000 --model-type "${typ}" > "logs/tok_${PRE}${p}_t${t}_${typ}.log" 2>&1 || return 1
    done
}

run_job () {  # $1 = gpu $2 = port $3 = kind $4 = model $5 = perm -> 0 done, 1 failed, 2 busy elsewhere
    local g=$1 port=$2 k=$3 m=$4 p=$5 rc=0
    local pr run t0 mb jlog
    pr=$(proto_of "${m}"); run=$(run_of "${k}" "${m}" "${p}"); t0=$(run_of t0 "${m}" "${p}")
    mb=${M_MBCAP[$m]:-32}; jlog="logs/models7_${run}.log"
    if [ -f "${R}/${run}/.complete" ]; then log "  skip ${run} (complete)"; return 0; fi
    live "${run}" && return 2
    case ${k} in
        distillm|amid|f12|f12sd)
            [ -f "${R}/${t0}/.complete" ] || [ "${DRY}" = "1" ] \
                || { log "  FAILED ${run}: no shared task0 ${t0}"; return 1; } ;;
    esac
    [ -e "${R}/${run}" ] && park "${run}"
    wait_gpu "${g}"
    log "  start ${run} on gpu${g}"
    [ "${DRY}" = "1" ] && return 0
    export MODEL_PATH=${M_PATH[$m]} MODEL_TYPE=${m} MODEL_TAG=${M_TAG[$m]} MB_CAP=${M_MBCAP[$m]}
    case ${k} in
        t0)
            ensure_tokenized "${m}" "${p}" || { log "  FAILED ${run}: tokenising, see logs/tok_${PRE}${p}_t*_${m}.log"; return 1; }
            PERM="${p}" GPU="${g}" PROTOCOL="${pr}" SEED="${SEED}" DATA_PREFIX="${PRE}" DIST_METHODS="" \
            MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh > "${jlog}" 2>&1 || rc=$? ;;
        inclora|migu)
            bash scripts/qwen/ced/run_cllora.sh --method "${k}" --data-root "data/${PRE}${p}" \
                --protocol "${pr}" --seed "${SEED}" --batch-size "${mb}" --grad-accum $((32 / mb)) \
                --eval-batch-size 16 --gpu "${g}" --py "${PY}" > "${jlog}" 2>&1 || rc=$? ;;
        distillm|amid)
            PERM="${p}" GPU="${g}" PROTOCOL="${pr}" SEED="${SEED}" DATA_PREFIX="${PRE}" DIST_METHODS="${k}" \
            MASTER_PORT="${port}" bash scripts/qwen/ced/dist_queue.sh > "${jlog}" 2>&1 || rc=$? ;;
        f12|f12sd)
            PERM="${p}" GPU="${g}" PROTOCOL="${pr}" SEED="${SEED}" DATA_PREFIX="${PRE}" \
            OURS_VARIANT=h2 OURS_SD=$([ "${k}" = f12sd ] && echo 1 || echo 0) RESUME=0 \
            MASTER_PORT="${port}" bash scripts/qwen/ced/ours_queue.sh --layers "${M_LAYERS[$m]}" \
                > "${jlog}" 2>&1 || rc=$? ;;
    esac
    if [ "${rc}" -eq 0 ] && [ -f "${R}/${run}/.complete" ]; then log "  done  ${run}"; return 0; fi
    log "  FAILED ${run} (exit ${rc}), see ${jlog}"
    return 1
}

JOBS=()   # kind:model:perm:phase, in hand-out order; a duplicate task0 entry is skipped once done
for ph in ${PHASES}; do
    for p in ${PERMS}; do
        case ${ph} in
            base) for m in ${MODELS}; do
                      for k in t0 inclora migu distillm amid; do JOBS+=("${k}:${m}:${p}:base"); done
                  done ;;
            ours) for m in ${OURS_MODELS}; do
                      for k in t0 f12 f12sd; do JOBS+=("${k}:${m}:${p}:ours"); done
                  done ;;
            *) log "unknown phase '${ph}' (base|ours)"; exit 1 ;;
        esac
    done
done

ready () {  # $1 = kind $2 = model $3 = perm $4 = phase: 0 when the job may start now
    local j k m p ph
    if [ "$4" = ours ]; then   # baselines first: nothing of Ours starts until every baseline job is over
        for j in "${JOBS[@]}"; do
            IFS=: read -r k m p ph <<< "${j}"
            [ "${ph}" = base ] && [ ! -e "${CLAIMS}/${k}_${m}_${p}.done" ] && return 1
        done
    fi
    case $1 in t0|inclora|migu) return 0 ;; esac
    [ -e "${CLAIMS}/t0_$2_$3.done" ]   # the shared task0 has been dealt with (done or failed)
}
worker () {  # $1 = gpu $2 = slot: take the first unclaimed ready job, then start over from the top
    local g=$1 s=$2 port=$((29700 + 10 * $1 + $2)) j k m p ph left rc n=0
    [ "${DRY}" = "1" ] || sleep $(( (s - 1) * 300 ))   # later slots see the earlier job's memory
    while :; do
        left=0
        for j in "${JOBS[@]}"; do
            IFS=: read -r k m p ph <<< "${j}"
            [ -e "${CLAIMS}/${k}_${m}_${p}.done" ] && continue
            left=1
            ready "${k}" "${m}" "${p}" "${ph}" || continue
            mkdir "${CLAIMS}/${k}_${m}_${p}" 2>/dev/null || continue   # atomic claim
            rc=0; run_job "${g}" "${port}" "${k}" "${m}" "${p}" || rc=$?
            if [ "${rc}" -eq 2 ]; then   # trained by another process: look again later
                rmdir "${CLAIMS}/${k}_${m}_${p}"; continue
            fi
            [ "${rc}" -eq 0 ] || n=$((n + 1))
            touch "${CLAIMS}/${k}_${m}_${p}.done"
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
log "=== all done: grep FAILED ${LOG}; results in ${R}/*_geneva_*_s${SEED}/ ==="
[ "${DRY}" = "1" ] || bash gather_logs.sh "geneva_models_$(date +%Y%m%d_%H%M)" || log "gather_logs.sh failed, run it by hand"
