# Adaptive Graph Vs OPSD

This branch, `adaptive_graph_vs_opsd`, contains only the adaptive graph redesign
and the files required to train, resume, test and evaluate it. Other methods and
experiment registries remain available in Git history and other branches.

## Code

- `adaptive_graph_ftrl_redesign/`: hard/soft teacher selection, graph masking,
  controller, virtual AdamW duel, checkpoints, configs, scripts and tests.
- [Algorithm](adaptive_graph_ftrl_redesign/ALGORITHM.md): equations and update order.
- [Method Guide](adaptive_graph_ftrl_redesign/README.md): controls and run examples.
- [Validation](adaptive_graph_ftrl_redesign/VALIDATION.md): checks and limitations.
- `eval/`: the existing math evaluator and result summarizer.
- `graph_based_scripts/run_graph_based_eval*.sh`: two retained evaluation launchers.

Legacy `adaptive_graph_ftrl_opsd`, `full_masked_graph_opsd`, `graph_based`,
`graph_free`, `se_opsd`, `se_opsd_lite`, `experiments` and `ablation` source
packages are no longer needed. Required helpers now live inside the redesign.
The FULL-only comparison remains available as `configs/full.yaml`.

## Environment

The locked GPU environment uses Python 3.13 and the existing dependency versions:

```bash
UV_PROJECT_ENVIRONMENT=dsd uv sync --extra cuda --group dev
source dsd/bin/activate
```

Reuse a working server environment when already installed. This cleanup does not
change the model, tokenizer, train/eval recipe or dependency versions. Training
requires a prepared operation-annotated hypergraph JSONL; datasets and model
weights are external inputs, not included in this branch.

## Train And Evaluate

Set `DATASET_JSONL` to your existing training JSONL. This example keeps the default
hard teacher, 8 duel controls, a 128-question heldout pool and 5 FULL warmup steps.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \\
RUN_ROOT=/research/cbim/vast/qt60/dsd/outputs \\
DATASET_JSONL=/path/to/hypergraph_openthoughts30k_operations_v3.jsonl \\
MAX_STEPS=75 WARMUP_STEPS=5 TRAIN_SEED=42 EVAL_SEED=42 \\
EVAL_STEPS=25,50,75 EVAL_DATASETS=aime24,aime25 \\
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh
```

Change parameters through environment variables or CLI, for example
`FTRL_RHO=0.8` or `--ftrl-rho=0.8`. The launcher creates a distinct run directory
with hyperparameters, seeds, timestamp and a config snapshot. See the method guide
for the full parameter table and resume behavior.

Standalone evaluation of existing checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m adaptive_graph_ftrl_redesign.evaluate \\
  --run-dir /path/to/run --steps 25,50,75 --datasets aime24,aime25 --seed 42
```

This preserves the existing graph-based vLLM evaluation recipe and legacy default
result location `full_masked_graph_opsd/outputs/evaluation/<run-name>/` for
compatibility. That directory contains results only, not a required Python
package. To choose result/cache storage explicitly, run `scripts/run_eval.sh`
inside the redesign package with `RUN_DIR`, `RESULTS_DIR` and `EVAL_WORK_DIR`.
Evaluation still needs writable storage; pointing these variables to another
filesystem does not delete old files.

## Offline Verification

```bash
python -m pytest -q
python -m adaptive_graph_ftrl_redesign.smoke_test
python -m adaptive_graph_ftrl_redesign.smoke_test --teacher-selection hard
```

Tiny tests do not download pretrained weights. CPU checks and mocked distributed
rollouts do not establish real CUDA/vLLM stability or AIME accuracy.
