# learn2follow server workflow

This is a source-only copy of the tracked working-tree files from
`dropout-self-distillation` (adaptive graph FTRL redesign), including configs,
tests, evaluation launchers, documentation, and the dependency lockfile.

## Run

1. Download Qwen3-8B using `download.txt`.
2. Download `dvtiendat/hypergraph_operations_openthoughts_30k` using `download.txt`.
   This supplies the required operation-annotated JSONL at the launcher's default
   path. Alternatively, set `DATASET_JSONL` to an existing server copy.
   Also download the three raw evaluation snapshots listed in `download.txt`.
3. From this folder, run:

```bash
bash project_commands.sh
```

Defaults: GPUs `2,3`, Qwen3-8B, hard teacher selection with final-answer
masking, 0 FULL warmup steps, 16 duel controls from a pool of 128,
100 updates, checkpoint evaluation at 25/50/75/100,
AIME25, AIME26 and HMMT25, training/evaluation seed 42, 12 samples per evaluation problem.
Training uses two workers with microbatch 1 and accumulation 16 (effective batch 32).
Evaluation uses tensor parallel size 2.
The original evaluator records Avg@12, Pass@12 and detailed results.

Before training, the launcher prepares local evaluation parquet files under
`data/raw/eval/{aime25,aime26,hmmt25}` into `data/processed/eval/` using OPSD's
`save_to_disk` layout. Training benchmark exclusion and evaluation read these
directories with `load_from_disk`; no benchmark is loaded from the Hub.
Override `RAW_DATA_ROOT` and `PREPARED_DATA_ROOT` to reuse existing local OPSD
data. Existing prepared datasets are preserved.

Example overrides:

```bash
MAX_STEPS=100 EVAL_DATASETS=aime25,aime26 bash project_commands.sh
```

Install the locked Python 3.13 CUDA environment separately under
`/mnt/local/uvenvs/learn2follow`. `project_commands.sh` activates that existing
environment and uses Python directly for training and evaluation.
Model/data/cache/output storage defaults to
`/mnt/local/aiskylimit_new_nothingnew_2/learn2follow`.
Each training launch creates a unique run directory under `outputs/`; the
original launcher logs its resolved path and writes `config.yaml` and
`train_eval.log` there.

The retained automatic evaluator writes results relative to this source folder:
`full_masked_graph_opsd/outputs/evaluation/<run-name>/`.

## Evaluate a saved run

From this folder after environment installation:

```bash
source /mnt/local/uvenvs/learn2follow/bin/activate
export EVAL_DATASET_DIR=/mnt/local/aiskylimit_new_nothingnew_2/learn2follow/data/processed/eval
CUDA_VISIBLE_DEVICES=2,3 python \
  -m adaptive_graph_ftrl_redesign.evaluate \
  --run-dir /path/to/saved/run --steps 25,50,75,100 \
  --datasets aime25,aime26,hmmt25 --seed 42 --launcher python
```

The annotated training input, model weights, and datasets are external assets.
See the copied `README.md` and method guide for algorithm settings and validation.

## Dependency comparison with ../opsd.txt

All 15 common pinned packages besides sentencepiece have matching versions,
including torch 2.8.0, transformers 4.57.1, TRL 0.26.0, vLLM 0.11.0,
accelerate 1.11.0 and datasets 3.6.0.

- Python: learn2follow requires 3.13; OPSD specifies 3.11.
- sentencepiece: learn2follow pins 0.2.1; OPSD pins 0.1.99.
- learn2follow additionally declares tqdm, wandb, pytest and ruff.
- OPSD additionally declares packaging, ninja, setuptools, wheel and pip
  without version pins; these are not explicit project requirements here.
- CUDA dependencies in learn2follow carry Linux/platform markers.

The environments are not identical. Keep learn2follow's original pyproject.toml
and uv.lock as the installation source rather than sharing the OPSD environment.
