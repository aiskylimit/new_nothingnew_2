# Offline GPT-2 +Contra with sampled timesteps

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
bash project_commands.sh train
```

The launcher uses existing tokenized datasets and formatted DistiLLM2 response
pairs. It pretrains the shared Dolly field on GPU 4 and the DistiLLM2 field on
GPU 5 before launching student training and evaluation. DistiLLM and FDD use GPU
4. DistiLLM2 and CSD use GPU 5. Each torchrun call starts one process.

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
| Generated DistiLLM2 paired data | `data/generated/distillm2/gpt2/t0/` |
| Downloaded teacher checkpoint | `results/gpt2/train/sft/gpt2-xlarge/` |
| Downloaded student initialization | `results/gpt2/train/init/gpt2-base/` |

Set `PROJECT_NAME`, `BASE_DIR`, or `ENV_DIR` if your provisioned root differs.
`TEACHER_PATH` and `STUDENT_PATH` can override checkpoint locations.
`DISTILLM2_DATA_DIR` can point to another directory of response pairs. The default
path reuses the pairs generated for the earlier experiment because pair generation
is independent of velocity-field training.

## Outputs

Velocity-field training samples a timestep uniformly from `[0, 1)` for each
example and reuses those timesteps across the layer pairs in a batch. Student
distillation queries the field at the student endpoint.

The launcher writes the shared field and projector to
`results/gpt2/train/velocity_field_sampled/` and the DistiLLM2 pair to its
`distillm2/` subdirectory. Student checkpoints go under
`results/gpt2/train/contra_sampled/`, evaluation outputs under
`results/gpt2/eval_main/`, and logs under `logs/gpt2_contra/`. These paths are
relative to the code directory.
