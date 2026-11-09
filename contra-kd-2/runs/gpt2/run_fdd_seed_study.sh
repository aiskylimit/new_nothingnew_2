#!/bin/bash

# Run through project_commands.sh to set up the offline environment.
set -e -o pipefail
set -m

BASE_PATH=$(cd "${1:-.}" && pwd)
cd "${BASE_PATH}"
TRAIN_ROOT="${BASE_PATH}/results/gpt2/train/fdd_seed_study"
EVAL_ROOT="${BASE_PATH}/results/gpt2/eval_main/fdd_seed_study"
LOG_DIR="${EVAL_ROOT}/logs"
LOCK_DIR="${TRAIN_ROOT}/.running"
export PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT}/fdd_seed_study"
export OPENWEBTEXT_TEXT_PATH="${PREPARED_DATA_ROOT}/fdd_seed_study/openwebtext/data.txt"
export PYTHONPATH=${BASE_PATH}
export PYTHONUNBUFFERED=1
export PET_STANDALONE=1
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

read -r -a GPUS <<< "${FDD_STUDY_GPUS:-4 5 6 7}"
if [[ ${#GPUS[@]} -ne 4 ]]; then
    echo "FDD_STUDY_GPUS must list four distinct GPU indices." >&2
    exit 2
fi
seen=" "
for gpu in "${GPUS[@]}"; do
    if [[ ! ${gpu} =~ ^(0|[1-9][0-9]*)$ || ${seen} == *" ${gpu} "* ]]; then
        echo "FDD_STUDY_GPUS must list four distinct GPU indices." >&2
        exit 2
    fi
    seen+="${gpu} "
done
JOBS_PER_GPU=${FDD_STUDY_JOBS_PER_GPU:-2}
if [[ ${JOBS_PER_GPU} != 1 && ${JOBS_PER_GPU} != 2 ]]; then
    echo "FDD_STUDY_JOBS_PER_GPU must be 1 or 2." >&2
    exit 2
fi

BENCHMARKS=(dolly self_inst vicuna sinst)
DATA_NAMES=(dolly self_inst vicuna sinst_11_)
DATA_DIRS=(dolly self-inst vicuna sinst/11_)
EVAL_SEEDS=(10 20 30 40 50)
MSE_SEED=1024

mkdir -p "${TRAIN_ROOT}" "${LOG_DIR}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Study is locked at ${LOCK_DIR}. Check whether its launcher is still running." >&2
    exit 1
fi

PIDS=()
JOB_NAMES=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    rmdir "${LOCK_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_job() {
    local name=$1 gpu=$2
    shift 2
    CUDA_VISIBLE_DEVICES=${gpu} "$@" >> "${LOG_DIR}/${name}.log" 2>&1 &
    PIDS+=("$!")
    JOB_NAMES+=("${name}")
    echo "Started ${name} on GPU ${gpu}."
}

wait_for_jobs() {
    local i pid status
    while [[ ${#PIDS[@]} -gt 0 ]]; do
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            if kill -0 "${pid}" 2>/dev/null; then
                continue
            fi
            if wait "${pid}"; then
                echo "Completed ${JOB_NAMES[$i]}."
                unset 'PIDS[i]'
            else
                status=$?
                unset 'PIDS[i]'
                echo "Failed ${JOB_NAMES[$i]} (exit ${status}). See ${LOG_DIR}/${JOB_NAMES[$i]}.log." >&2
                return "${status}"
            fi
        done
        if [[ ${#PIDS[@]} -gt 0 ]]; then
            sleep 1
        fi
    done
    JOB_NAMES=()
}

for checkpoint in "${STUDENT_PATH}" "${TEACHER_PATH}"; do
    if [[ ! -s ${checkpoint}/config.json ]]; then
        echo "Missing downloaded checkpoint: ${checkpoint}/config.json" >&2
        exit 1
    fi
done
for input in dolly/raw.jsonl dolly/valid.jsonl self-inst/valid.jsonl vicuna/valid.jsonl sinst/11_/valid.jsonl; do
    if [[ ! -s ${RAW_DATA_ROOT}/${input} ]]; then
        echo "Missing downloaded input: ${RAW_DATA_ROOT}/${input}. Provision download.txt first." >&2
        exit 1
    fi
done

if [[ ! -s ${PROCESSED_DATA_ROOT}/.complete ]]; then
    echo "Preprocessing OpenWebText and Dolly for the study."
    if [[ ! -s ${OPENWEBTEXT_TEXT_PATH} ]]; then
        CUDA_VISIBLE_DEVICES="" python3 -u tools/get_openwebtext.py \
            2>&1 | tee "${LOG_DIR}/prepare_openwebtext.log"
    fi
    CUDA_VISIBLE_DEVICES="" python3 -u tools/process_data_pretrain.py \
        --data-dir "$(dirname "${OPENWEBTEXT_TEXT_PATH}")" \
        --processed-data-dir "${PROCESSED_DATA_ROOT}/openwebtext/gpt2/512" \
        --model-path "${STUDENT_PATH}" --model-type gpt2 --max-length 512 \
        --train-num 22870 --data-process-workers "${DATA_PROCESS_WORKERS}" --dev-num 1000 \
        2>&1 | tee "${LOG_DIR}/preprocess_openwebtext.log"
    CUDA_VISIBLE_DEVICES="" python3 -u tools/process_data_dolly.py \
        --data-dir "${RAW_DATA_ROOT}/dolly" --processed-data-dir "${PROCESSED_DATA_ROOT}/dolly/full" \
        --model-path "${STUDENT_PATH}" --model-type gpt2 --max-prompt-length 256 \
        --data-process-workers "${DATA_PROCESS_WORKERS}" --dev-num 1000 \
        2>&1 | tee "${LOG_DIR}/preprocess_dolly.log"
    printf 'complete\n' > "${PROCESSED_DATA_ROOT}/.complete"
fi

run_student() {
    local method=$1 seed=$2 port=$3 i eval_seed metrics train_script
    local ckpt="fdd_seed_study/${method}/${seed}"
    local student_path="${BASE_PATH}/results/gpt2/train/${ckpt}"
    local field_path="${TRAIN_ROOT}/fields/${seed}"
    local extra_args=()

    if [[ ! -s ${student_path}/.complete ]]; then
        train_script="fdd_base.sh"
        if [[ ${method} == contra_fdd ]]; then
            train_script="contra_base.sh"
            if [[ ! -s ${field_path}/.complete ]]; then
                bash scripts/gpt2/train_velocity_field.sh "${BASE_PATH}" "${port}" 1 \
                    --seed "${seed}" --save "${field_path}"
                test -s "${field_path}/projector.pth"
                test -s "${field_path}/velocity_field.pth"
                printf 'complete\n' > "${field_path}/.complete"
            fi
            extra_args=(--velocity-field-path "${field_path}/velocity_field.pth"
                        --projector-path "${field_path}/projector.pth" --save-velocity-checkpoints)
        fi
        bash "scripts/gpt2/fdd/${train_script}" "${BASE_PATH}" "$((port + 1))" 1 \
            --seed "${seed}" --save "${student_path}" "${extra_args[@]}"
        test -s "${student_path}/config.json"
        if [[ ${method} == contra_fdd ]]; then
            test -s "${student_path}/projector.pth"
            test -s "${student_path}/velocity_field.pth"
            test -s "${student_path}/velocity_checkpoint.json"
        fi
        printf 'complete\n' > "${student_path}/.complete"
    fi

    for i in "${!BENCHMARKS[@]}"; do
        for eval_seed in "${EVAL_SEEDS[@]}"; do
            metrics="${EVAL_ROOT}/generation/${method}/${seed}/${DATA_NAMES[$i]}-512/${eval_seed}/metrics.json"
            if [[ ! -s ${metrics} ]]; then
                bash "scripts/gpt2/eval/eval_main_${BENCHMARKS[$i]}.sh" "${BASE_PATH}" "$((port + 2))" 1 "${ckpt}" \
                    --ckpt-name "${method}/${seed}" --save "${EVAL_ROOT}/generation" \
                    --seed "${eval_seed}" --eval-batch-size 64
                test -s "${metrics}"
            fi
        done
    done
}

# Each pipeline keeps its assigned GPU through field pretraining, training, and generation evaluation.
start_job contra_fdd_1024 "${GPUS[0]}" run_student contra_fdd 1024 "$((BASE_PORT + 0))"
start_job contra_fdd_2048 "${GPUS[1]}" run_student contra_fdd 2048 "$((BASE_PORT + 3))"
start_job contra_fdd_4096 "${GPUS[2]}" run_student contra_fdd 4096 "$((BASE_PORT + 6))"
start_job fdd_1024 "${GPUS[3]}" run_student fdd 1024 "$((BASE_PORT + 12))"
if [[ ${JOBS_PER_GPU} == 1 ]]; then
    wait_for_jobs
fi
start_job fdd_4096 "${GPUS[2]}" run_student fdd 4096 "$((BASE_PORT + 9))"
start_job fdd_2048 "${GPUS[3]}" run_student fdd 2048 "$((BASE_PORT + 15))"
wait_for_jobs

for i in "${!BENCHMARKS[@]}"; do
    mse_path="${EVAL_ROOT}/layer_changes/${MSE_SEED}/${BENCHMARKS[$i]}"
    if [[ ! -s ${mse_path}/layer_changes.json ]]; then
        contra_path="${TRAIN_ROOT}/contra_fdd/${MSE_SEED}"
        start_job "layer_changes_${BENCHMARKS[$i]}" "${GPUS[$i]}" \
            torchrun --standalone --nproc-per-node 1 evaluate_layer_changes.py \
                --model-path "${contra_path}" --baseline-model-path "${TRAIN_ROOT}/fdd/${MSE_SEED}" \
                --teacher-model-path "${TEACHER_PATH}" --projector-path "${contra_path}/projector.pth" \
                --seed "${MSE_SEED}" --data-names "${DATA_NAMES[$i]}" --data-dir "${RAW_DATA_ROOT}/${DATA_DIRS[$i]}" \
                --json-data --max-prompt-length 256 --max-length 512 --num-workers 0 --eval-batch-size 8 \
                --save "${mse_path}"
    fi
done
wait_for_jobs

python3 tools/summarize_fdd_study.py "${EVAL_ROOT}"
echo "Study complete. Results: ${EVAL_ROOT}/summary.json"
