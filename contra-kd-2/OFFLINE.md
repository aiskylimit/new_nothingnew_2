# Offline GPT-2 +Contra

The original contra-kd checkout is unchanged. This copy contains source code,
not the original `data/`, Git metadata, environments, caches or logs.
Remote artifact publication and external experiment reporting have been removed
from the entire copy, including legacy scripts, dependencies, credentials and
configuration. Existing console and filesystem logs remain available.

## Provision before entering the offline machine

1. Use `../contra.txt` with the environment provisioner, like `opsd.txt`.
   Python 3.11 environment: `contra`, default `/mnt/local/uvenvs/contra`.
   For NVIDIA B200, provision PyTorch 2.8.0 with CUDA 12.8 and DeepSpeed 0.18.2.
   CPU-only or older CUDA builds are not suitable for this GPU. The host NVIDIA
   driver must also support this CUDA runtime. Rebuild the environment after
   changing contra.txt; editing the file does not update an existing environment.
2. Use `download.txt` with the external downloader, like `opsd/download.txt`.
   It downloads complete snapshots of:
   - `Elriggs/openwebtext-100k`
   - `MiniLLM/dolly` (raw training records and held-out evaluation)
   - `MiniLLM/self-inst`
   - `MiniLLM/Vicuna`
   - `MiniLLM/sinst` (evaluation uses `11_/valid.jsonl`)
   - Teacher: `bachthetrollface/gpt2-1.5B-teacher-dolly`
   - Student init: `bachthetrollface/gpt2-120M-init-dolly`

No datasets from the original checkout are copied or used. The pipeline never
downloads resources or installs packages on the offline machine.

## Export saved evaluation results

```bash
bash project_commands.sh
```

The launcher copies `results/gpt2/eval_main/` into its `results/` subdirectory.
The copy preserves relative paths and skips `preds.txt` and `answers.jsonl`.
Each run replaces the previous export with a fresh copy and prints its size.

## Multi-seed study

The training launcher is retained as comments in `project_commands.sh`.
The study trains FDD and FDD + CONTRA at seeds `1024`, `2048`, and
`4096`. The `--seed` argument controls Python, NumPy and PyTorch randomness,
including training data order in fine-tuning, field pretraining and field
updates. The launcher preprocesses Dolly and OpenWebText once. Each CONTRA run
pretrains its own projector and velocity field, then saves the active projector
and field with the selected student checkpoint.

| Physical GPU | Student pipelines |
| --- | --- |
| 4 | FDD + CONTRA, seed 1024 |
| 5 | FDD + CONTRA, seed 2048 |
| 6 | FDD + CONTRA and FDD, seed 4096 |
| 7 | FDD, seeds 1024 and 2048 |

Each pipeline keeps its GPU through training and generation evaluation. Two
pipelines run concurrently on GPUs 6 and 7. `FDD_STUDY_GPUS` can override the
four physical indices as a space-separated list. Set `FDD_STUDY_JOBS_PER_GPU=1`
to run one student pipeline per GPU at a time. Each student runs generation
evaluation on Dolly, Self-Instruct, Vicuna and SInst with seeds
`10 20 30 40 50`. After the six pipelines finish, the four GPUs score one
dataset each for the seed-1024 layer-change comparison. Both students use the
projector saved with the selected FDD + CONTRA student checkpoint.

Study checkpoints are under `results/gpt2/train/fdd_seed_study/`. The text-only
outputs are under `results/gpt2/eval_main/fdd_seed_study/`, including generation
scores, predictions, per-transition MSE, job logs, and `summary.json`. Rerunning
the study reuses completed preprocessing, training and evaluation stages.
Incomplete training restarts from the initial student checkpoint. Use a fresh
output directory if you change the study inputs.

## Dolly one-step evaluation

```bash
bash project_commands.sh one-step-dolly
```

This uses the saved seed-1024 FDD + CONTRA student, projector, and velocity
field. It projects each aligned student layer into teacher space, takes a full
Euler step from time zero, and compares the result with the corresponding
teacher layer on Dolly reference-response tokens. The per-layer and overall
MSE, cosine similarity, and cosine distance are saved to
`results/gpt2/eval_main/fdd_seed_study/one_step/1024/dolly/one_step.json`.
The job uses GPU 4 by default. Set `ONE_STEP_GPU` to select a different GPU.

## Original four-method workflow

The retained training launcher accepts the `contra` argument for this workflow.

CPU preprocessing hides all GPUs. Generation runs on physical GPU 4, followed
by the shared Dolly field on GPU 4 and the DistiLLM2 field on GPU 5. Once both
fields are ready, DistiLLM and FDD run on GPU 4 and DistiLLM2 and CSD run on GPU 5.
Each training job runs its evaluations on the same GPU. Every torchrun call
starts one process and sees its assigned physical GPU as logical CUDA device 0.

## Path contract

`@PROJECT@` in `download.txt` must resolve to `PROJECT_NAME` in the launcher,
default `aiskylimit_new_nothingnew_2`. `BASE_DIR` defaults to
`/mnt/local/aiskylimit_new_nothingnew_2/contra-kd-2`. Provision the assets at
that path before launching.

| Data | Path relative to BASE_DIR |
| --- | --- |
| Downloaded OpenWebText snapshot | `data/raw/openwebtext-100k/data/train-*.parquet` |
| Downloaded Dolly training records | `data/raw/dolly/raw.jsonl` |
| Downloaded Dolly evaluation | `data/raw/dolly/valid.jsonl` |
| Downloaded Self-Instruct evaluation | `data/raw/self-inst/valid.jsonl` |
| Downloaded Vicuna evaluation | `data/raw/vicuna/valid.jsonl` |
| Downloaded SInst evaluation | `data/raw/sinst/11_/valid.jsonl` |
| Generated Dolly generation prompts | `data/prepared/dolly/{train,dev}.jsonl` |
| Converted OpenWebText text | `data/prepared/openwebtext/data.txt` |
| Tokenized Dolly | `processed_data/dolly/full/gpt2/` |
| Tokenized OpenWebText | `processed_data/openwebtext/gpt2/512/22.87K/` |
| Generated DistiLLM2 paired data | `data/generated/distillm2/gpt2/` |
| Downloaded teacher checkpoint | `results/gpt2/train/sft/gpt2-xlarge/` |
| Downloaded student initialization | `results/gpt2/train/init/gpt2-base/` |

All active preprocessing, generation, field, training and evaluation scripts
use the exported absolute paths. The seed study puts prepared data under
`data/prepared/fdd_seed_study/` and tokenized data under
`processed_data/fdd_seed_study/`. The code directory may be separate from
`BASE_DIR`. Set `PROJECT_NAME`, `BASE_DIR` or `ENV_DIR` before launching if your
provisioned root differs.
TEACHER_PATH and STUDENT_PATH may override model locations explicitly.

MiniLLM/dolly has raw.jsonl and valid.jsonl but no train/dev files. The launcher
creates prompts from the downloaded raw.jsonl: first 1,000 records for dev,
remaining records for train, matching process_data_dolly.py. Evaluation uses
the independently downloaded valid.jsonl, never these training/dev files.
OpenWebText conversion preserves the original newline replacement and 250,000
record cap (the selected snapshot contains 100,000 records).

Model generation uses the existing Transformers fallback; vLLM is omitted to
avoid conflicting with the training Transformers pin. The environment uses
Transformers 4.57.1 and Tokenizers 0.22.1 to read the checkpoint tokenizer format.
Field pretraining and periodic field updates sample times uniformly from
`[0, 1)`. Offline HF flags are enabled. No NLTK download is needed.

Original workflow outputs are relative to the **code directory**. Fields are written to
`results/gpt2/train/velocity_field/` and its `distillm2/` subdirectory. Student
checkpoints go to `results/gpt2/train/contra/`, evaluation results to
`results/gpt2/eval_main/`, and logs to `logs/gpt2_contra/`. Re-running repeats
generation and training using the same output paths.

Validation is static plus mocked end-to-end shell execution. No full dependency
installation, actual model/data provisioning or CUDA training is performed here.
