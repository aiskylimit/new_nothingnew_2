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

## Run

```bash
bash project_commands.sh
```

CPU preprocessing hides all GPUs. Generation runs on physical GPU 4, followed
by the shared Dolly field on GPU 4 and the DistiLLM2 field on GPU 5. Once both
fields are ready, DistiLLM and FDD run on GPU 4 and DistiLLM2 and CSD run on GPU 5.
Each training job runs its evaluations on the same GPU. Every torchrun call
starts one process and sees its assigned physical GPU as logical CUDA device 0.

## Path contract

`@PROJECT@` in `download.txt` must resolve to `PROJECT_NAME` in the launcher,
default `aiskylimit_new_nothingnew_2`. `BASE_DIR` defaults to
`/mnt/local/aiskylimit_new_nothingnew_2/contra-kd`.

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
use the exported absolute paths. There is no fallback to the old checkout's
`data/`. The code directory may be separate from BASE_DIR. Set `PROJECT_NAME`,
`BASE_DIR` or `ENV_DIR` before launching if your provisioned root differs.
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

Outputs are relative to the **code directory**. Fields are written to
`results/gpt2/train/velocity_field/` and its `distillm2/` subdirectory. Student
checkpoints go to `results/gpt2/train/contra/`, evaluation results to
`results/gpt2/eval_main/`, and logs to `logs/gpt2_contra/`. Re-running repeats
generation and training using the same output paths.

Validation is static plus mocked end-to-end shell execution. No full dependency
installation, actual model/data provisioning or CUDA training is performed here.
