# Offline SDXL Q3/DSPO on 2xB200

The platform entrypoint is:

```bash
bash project_command.sh
```

The repository-level `commands.sh` first measures real peak memory across
10-step candidates on physical GPUs 2 and 3, selects the largest micro-batch
at or below 95% VRAM, and runs a short end-to-end pilot. After that pilot
passes, start the full 851,293-pair train/infer/eval pipeline with:

```bash
bash hessian/run_b200_full.sh
```

It sources `env.sh`, activates the uv environment created by the platform from
the root-level `sdxl-q3-offline-b200-2gpu.txt`, validates all local assets and two scheduler-assigned
Blackwell GPUs, launches Accelerate DDP,
evaluates the final model immediately, and writes `results/summary.json`.

The environment specification lives outside this source directory at the
repository root. Its filename exactly matches the environment directory:
`source /mnt/local/uvenvs/sdxl-q3-offline-b200-2gpu/bin/activate`.
The platform must also process this project's `download.txt` manifest before
the job starts. It downloads every model and dataset into `offline_assets/`;
the runtime itself sets all Hugging Face libraries to offline mode and
disables external reporting/upload.

The OpenAI `clip` Python module and tokenizer vocabulary are vendored under
`clip/`. The environment specification intentionally contains no `git+https`
dependency, so uv never needs to contact GitHub.

Default production semantics:

- GPUs: physical `2,3` by default
- micro-batch: selected on the B200 node by the 95%-VRAM autotuner
- gradient accumulation: 1 after tuning
- effective batch: `2 * selected micro-batch`
- full dataset: 851,293 binary preference pairs, exactly one pass
- optimizer steps: `ceil(851293 / effective_batch)`
- precision: BF16
- model: SDXL 1.0 at 1024x1024
- method: Q3 (TBPO + reference-MSE policy weighting + DSPO winner anchor)

Detailed logs are in `runtime/logs/project.log`, per-run training logs are in
`runtime/runs/<run>/console.log`, and per-stage evaluation logs are in
`runtime/eval/<eval>/logs/`. Rerunning the entrypoint reuses completed model,
prompt, image, score, and report artifacts.

For a non-platform pilot, point `VENV_DIR` at an existing compatible uv
environment. For example:

```bash
PIPELINE_MODE=pilot GPU_IDS=2,3 NUM_GPUS=2 TARGET_GPU_FAMILY=A100 \
  VENV_DIR=/path/to/existing/venv \
  OFFLINE_EVAL_LIMIT=2 bash project_command.sh
```
