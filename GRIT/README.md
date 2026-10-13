# GRIT

GRPO safety training that keeps the base model's general capability with two
protections: a null-space projection of the update (NSPO-style, per MLP layer)
and a KL trust region anchored to the frozen base model. This folder runs it
on **Qwen2.5-3B-Instruct** with **Qwen3Guard-Gen-4B** as the reward model on
**one B200 (180 GB)**, on the offline server. **To run it, follow [RUN.md](RUN.md).** See [PLAN.md](PLAN.md) for the
method, the fixed configuration and the design decisions.

## Run

1. Environment: `../grit.txt` (uv, Python 3.11) builds `/mnt/local/uvenvs/grit`.
2. Download (`#d` job): the contents of [download.txt](download.txt). Models, the
   pre-built GRIT artifacts (`fushinguyenex/GRIT-artifacts`, ~5 GB) and the eval
   benchmarks. Nothing is downloaded at run time and no HF token is used.
3. Run (`#1` job): `cd ./GRIT && bash ./project_commands.sh`

| Stage | What it does |
|---|---|
| `check` | versions, GPU, free VRAM, CPU regression tests |
| `verify` | sha256 of every artifact, manifests (float32, top-64, revision), tokenizer fingerprint, base-logit spot check |
| `dryrun` | 3 training steps with one validation and one checkpoint; picks the first memory config in `MEM_CONFIGS` that passes (skipped once it has passed) |
| `train` | 40 GRIT steps, checkpoint and validation every 5 steps; resumes from the latest checkpoint and falls back to a less aggressive memory config on failure |
| `eval` | base + checkpoints (`EVAL_STEPS`, default `20 40`) evaluated in parallel (`EVAL_SLOTS` vLLM engines per GPU): safety (PKU-SafeRLHF 1,000, XSTest unsafe, OR-Bench toxic: unsafe rate), over-refusal (XSTest safe, OR-Bench hard-1k, AlpacaEval: refusal rate on benign prompts), MMLU, SuperGPQA, GSM8K, MATH-500, OlympiadBench, LiveCodeBench |
| `summary` | training curve + eval table + paired deltas vs base, printed and saved |

`commands.sh` runs only `cd ./GRIT` + `bash ./project_commands.sh`. Settings are the
defaults at the top of `project_commands.sh`: `GPU` (`auto` = the GPU with the most free
memory), `RUN_ID`, `LAMBDA_PRES` (default 1; 0 is the baseline), `MAX_STEPS`, `STAGES`
(comma list of the stages above), `EVAL_STEPS`, `EVAL_MAX_SAMPLES`, `MEM_CONFIGS` /
`MEM_CONFIG`, `EVAL_GPUS`, `EVAL_SLOTS`. When running by hand, the same names work as
env vars.

## GPU memory

- **Train:** colocated actor + vLLM + Qwen3Guard on one GPU. With vLLM sleep mode
  (`--rollout-sleep`), vLLM releases all its GPU memory (level 2) after each rollout,
  so the actor's backward and curvature phases get that memory back. The next sync
  reloads the new weights anyway. A memory config is
  `TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP`. The default list is
  `16:4:64:0.40:1 8:2:32:0.40:1 8:2:32:0.30:0`, and the last entry is the original setting.
- **Eval:** several vLLM engines share each GPU (memory split evenly), with startups
  serialized per GPU so memory profiling never overlaps. Each engine runs up to 512
  concurrent sequences.


## Outputs

- `/mnt/local/aiskylimit_new_nothingnew_2/grit/outputs/<RUN_ID>/step_XXXXXX`:
  checkpoints (weights for every saved step; optimizer state only in the latest).
- `/mnt/local/aiskylimit_new_nothingnew_2/grit/eval/<RUN_ID>/`: raw generations and guard labels.
- `GRIT/results_<RUN_ID>/`: small files to fetch with a `#2` job: `logs/`,
  `train/metrics.jsonl`, `train/diagnostics_summary.json`, `train/validation/`,
  `eval/eval_results.json`, `summary.md`, `summary.json`.

## Code

| Path | Content |
|---|---|
| `grit/` | GRIT math: GRPO objective, AdamW predictor, projection, trust region, curvature (HVP), persistent vLLM rollout |
| `scripts/train_grit.py` | trainer (FP32 actor, colocated vLLM, local checkpoints only) |
| `scripts/verify_artifacts.py` | offline verification of the pre-built artifacts |
| `scripts/eval_run.py`, `eval_worker.py`, `eval_lib.py` | evaluation (vLLM generation, Qwen3Guard judging, scoring, paired bootstrap) |
| `scripts/summarize_results.py` | console + markdown summary |
| `third_party/` | official scorers of OlympiadBench and LiveCodeBench, vendored unchanged (MIT), see `third_party/README.md` |
| `tests/` | CPU regression tests and math checks (`python -m pytest -q`) |

The code never uploads anything: there is no Hub push, no external judge API
and no credential handling.
