# Adaptive Graph FTRL Redesign

For the implemented hard/soft algorithms, equations, update order and a matched
FULL-only comparison, see [ALGORITHM.md](ALGORITHM.md). The dedicated experiment
branch is `adaptive_graph_vs_opsd`.

This branch contains the self-contained redesign and its train/eval dependencies.
Legacy adaptive, FULL, SE, graph-free, experiments and ablation packages have
been removed from this branch; their history remains on other branches.
The matched FULL-only baseline is `configs/full.yaml` in this package.
This is an experimental method, not evidence of improved accuracy.

## Recipe

- Qwen3-4B, BF16 frozen backbone, FP32 LoRA rank 64 / alpha 128, dropout off.
- Current-round self-teachers: thinking enabled; student rollouts non-thinking.
- Train/eval base seeds both 42. Four ranks, microbatch 1, accumulation 8,
  effective batch 32. One question-only student completion/example, batched vLLM.
- Train sampling temperature 1.1, top-p .95, top-k -1, max new tokens 1024.
  Full-vocabulary forward KL also uses temperature 1.1, FP32 chunk size 64,
  per-example token mean followed by example mean. No token-loss clipping.
- AdamW LR 5e-6, betas .9/.999, epsilon 1e-8, weight decay 0,
  combined-gradient clipping .1, continuous linear LR decay.
- 100 optimizer steps by default. Steps 1-5: FULL only, effective weights 1/0,
  latent weights .5/.5. No action, MASK scoring, controls, reference scoring,
  virtual branches or adaptive-statistic updates. Optimizer/scheduler do not reset.
- Steps 6-10: first adaptive window, old weights .5/.5. Feedback after step 10
  affects steps 11-15. H=5; there are 19 adaptive decisions at a 100-step endpoint.
  The first three valid actor windows initialize statistics, not the actor.
  Earliest actor update is after step 25, assuming all actor observations valid.
- Alpha is retained reasoning exposure, not percent masked. Ten-parameter linear
  Beta controller starts exactly Beta(2,2), Adam .01, grad clip 1. No MLP,
  node scorer, grader, third teacher, residual projection or hard-zero gate.

## Graph And Context

The balanced-brace terminal-answer parser and graph candidate builder live in
`data.py`, `schema.py` and `masks.py`. Answer splitting happens before graph creation;
answer-only nodes are removed and suspicious outgoing dependencies rejected.
FULL preserves the original reference. MASK keeps original selected reasoning
fragments and, by default, an explicitly labeled `Final answer:` block.

Candidates are descendant-closed single-node closures, unions of two closures,
and all-hidden, deduplicated with no empty mask. This finite pool is not all
possible masks. Pair cap 4096 uses stable example-specific sampling. Sampling
chooses a unique retained-token level proportional to
`exp(-abs(exposure-alpha)/.05)`, then a uniform mask within that level.
Only the selected MASK is scored. Positive node token costs are required.

`mask_teacher_include_final_answer: false` removes the explicit answer block
from MASK, not intermediate values that may reveal it. FULL still has its answer.
All-hidden is answer-only with the flag true and question/template-only otherwise.
Branching, candidate counts, unique levels, mask diversity and exact context
aliases are logged. Main context overflow fails explicitly, never silently
truncates a teacher or falls back to the raw reference.

## Separate Feedback Channels

The following describes the default `teacher_selection: soft`. The optional
hard mode below keeps graph/controller behavior but changes the student target
and the expert feedback cadence.

### Main Update

FULL and MASK score the same generated continuation at the same pre-step snapshot.
Detached targets are mixed as probabilities `q = w_F*q_F + w_M*q_M`.
The student learns `KL(q || p)`. Old weights stay fixed throughout the window.
Last-step pure gradients obey `g_mix = w_F*g_F + w_M*g_M`; combining them avoids
a third backward, with exactly the same raw gradient before clipping.

### Direct-Answer FTRL Control

The fixed 128-question labeled pool is disjoint from main questions and AIME24/25.
Shuffle/visit without replacement until exhausted, using its own RNG. Freeze
16 controls/window before action; at least 8 must tokenize and fit the context.
There is **no control generation and no boxed/cue requirement on a student output**.

Input is question + `Return only the final answer, without reasoning, inside
\boxed{}.` in a non-thinking user template, followed by the literal cue
`\nFinal answer: \boxed{` (real newline, single LaTeX backslash at runtime).
Labels are canonical gold answer-content tokens: a boxed wrapper/heading is
unwrapped; malformed boxes are reported invalid. Only label tokens are scored,
with standard causal shift and answer temperature 1.0. No reference reasoning,
gold label before its prediction position, or control gradient in actual training.

At the last pre-update snapshot, pure FULL/MASK gradients use the **entire last
main batch**, accumulated microbatch-wise. Two isolated `torch.optim.AdamW`
instances receive cloned trainable LoRA weights and the same current optimizer
state, LR, betas, epsilon, weight decay and clipping rule. No approximate SGD
branch, virtual LR multiplier, live optimizer mutation or scheduler step.
Temporarily install each branch's LoRA to score fixed controls under no-grad/eval.
Restore original LoRA, buffer values/bindings, modes and RNG, including failures.
The frozen backbone is shared, not cloned. Live `.grad` and optimizer state are
never touched by virtual branches. Record three means `L_0`, `L_F`, `L_M`.

`delta = (L_F-L_M)/(L_0+1e-12)`, positive favors MASK.
Historical RMS uses EMA .9, floor 1e-4, initially the floor, **without centering**.
Before ingesting this observation:

```text
scaled_delta = clip(delta / max(historical_RMS, 1e-4), -1, 1)
D_next = .9*D + scaled_delta
w_M_next = .02 + .96*sigmoid(3*D_next)
w_F_next = 1-w_M_next
```

Tiny/zero contrast, FP32 loss resolution, raw/scaled delta and floor activation
are diagnostics, not a reason to amplify virtual LR. An invalid duel freezes
D, weights, delta statistics and last-valid delta, without decay. It does not
invalidate an otherwise valid actor reward. Controls are adaptive validation,
not an untouched test set; direct-answer NLL is not end-to-end reasoning accuracy.

### Hard Teacher Every Step

Use `--teacher-selection hard` or `configs/graph_hard_answer.yaml`. Default soft
configs and their eta=3 setting are unchanged. In hard mode, **one teacher is
selected for the entire optimizer step**, including all accumulation microbatches:

```text
D = 0; teacher = FULL
q_train = q_FULL if teacher == FULL else q_MASK
g_train = g_FULL if teacher == FULL else g_MASK

# Every adaptive optimizer step, measure the existing exact AdamW virtual duel
# at the pre-update snapshot, then commit the actual selected-expert update.
scaled_delta = clip(delta / max(historical_RMS, 1e-4), -1, 1)
D_next = rho * D + scaled_delta
teacher_next = MASK if D_next > 0 else FULL
```

The target and actual gradient are selected directly, not averaged. FULL wins
ties. There is no sigmoid, mixture floor, exploration weight or permanent
teacher lock; a previously unselected teacher can win the next valid duel.
`ftrl_eta` and `ftrl_min_weight` apply only to soft mode. This is a **discounted
hard leader on RMS-normalized feedback**, not classical undiscounted FTL on raw
loss sums. It retains `rho=0.9` and the existing RMS feedback by default.
An invalid duel leaves the leader, D and delta statistics unchanged, without
decay. `fixed_mixture` cannot be combined with hard selection.

Both teachers still score the same question-only rollout and supply pure
gradients for each duel; only the selected gradient updates the student. The
new leader takes effect on the **next** optimizer step. Alpha, the frozen 16
controls and the exposure-controller policy stay fixed over H=5 steps. Actor
updates keep the three-valid-window warmup and window-end cadence. Since hard
choices can change inside a window, its advantage uses the **realized fraction
of steps trained by MASK** times the historical standardized actor reward,
instead of the start-of-window weight. This frequency is not a soft training
mixture. Actor reward is still a noisy proxy; teacher switches also change the
loss objective across observations.

FULL-only warmup remains five steps unless explicitly overridden with
`--full-opsd-warmup-steps 0`. A 50-step run therefore attempts 45 per-step duels
after warmup, instead of nine in soft mode. Expect extra pure-gradient and
control-scoring cost. Frozen-target replay stays window-end only. Per-step
logs include `teacher`, `teacher_next`, `delta`, `D`, `ftl_updated` and skip
reason; JSON metrics include the full `expert_feedback`. Window logs and
checkpoints preserve every step's selected weight and duel feedback, including
mid-window leader changes. Resume across soft/hard modes is rejected; older
soft checkpoints without the new selection field remain compatible.

### Exposure Actor Reward

Loss observations are this window's pre-update main mixture losses, plus one
post-last-update replay loss on the same last batch, frozen targets and old weights.
There are H+1 losses for H actual steps; no previous-window/warmup loss enters.

At each adaptive step, independently select 4 main examples using a separate
checkpointed RNG, independent of alpha/mask outcomes. Reuse their selected MASK
and pre-step teacher snapshot to teacher-force verified reference labels; do not
generate extra teacher/reference/control completions. Average log-probability
over label tokens, then examples, at the same teacher temperature 1.1.
Suffix scoring is causally shifted; privileged MASK context remains allowed to
contain retained reference fragments and optional answer.

Declared implementation choice: reference label cap **1024 tokens**. When needed,
keep a prefix of the original reasoning followed by the complete original answer
block. Capacity is also bounded by the longest unmasked rendering, computed
independently of the chosen mask and answer-visibility flag. If even the complete
answer cannot fit, the grounded observation is invalid. Log original/scored token
counts, retained fraction and truncation. This is a **truncated-reference score**
when capped, not a claim of full-reference coverage. Full-fit references are
scored verbatim. Both answer-visibility variants have identical label spans.

```text
G_lp = sum_i .95**(i-1) * max(0, ell[i-1]-ell[i])
G_gt = sum_i .95**(i-1) * reference_logprob[i] / sum_i .95**(i-1)
R = G_lp + .01*G_gt
A = w_M_used * historical_Z_R(R)
actor_loss = -detach(A) * log_prob(detach(alpha) | detach(saved_state))
```

No fast-duel bonus in the actor reward. Delta only enters FTRL and lagged state.
State is `[scheduled_time, last_valid_Z_delta, old_w_M, last_valid_Z_R]`;
reward/delta historical EMA standardizers have variance floors 1e-4/1e-8 and
clip [-5,5]. Current signals are ingested after using historical statistics.
The actor has a separate Beta-sampling RNG, so control/action sampling does not
consume main-rollout RNG or main data order. Invalid actor observations freeze
reward stats/warmup/last reward, not a valid duel. All-alias graph observations
are excluded. Three valid actor rewards warm up statistics; FTRL can update on
the first valid duel. Frozen actor still scores grounding and learns FTRL.

This is the specification's ATESD-style H=L=5 proposal, not a replication of a
longer-lookahead paper experiment. Intermediate batches/teacher targets change;
clipped decreases are noisy progress proxies. Reference copying/exposure bias
remains. Neither R nor delta proves correctness or causal downstream contribution.

## Memory, Logs And Resume

No full-vocabulary targets from earlier steps are retained. Last-batch frozen
mixture targets are streamed as FP32 chunks to an ephemeral disk cache, then
replayed one example at a time and deleted. This preserves the full-batch replay
without retaining 32 vocabulary tensors in RAM. `--replay-cache-dir` can select
fast local scratch. Allow roughly `local_batch * completion_tokens * vocab * 4`
bytes **per rank** (about 4.6 GiB for 8x1024x151936), plus file overhead. Disk I/O
is real overhead: cache bytes and write time are logged, not claimed free.

Like `se_opsd`, rank zero displays one standard dynamic tqdm progress bar and
writes one dictionary of metrics after every optimizer step: loss, grad norm,
LR, phase, alpha, token/step mask percentages, grounded score and effective F/M.
One compact feedback dictionary per adaptive window adds R, raw/scaled delta,
next weight and independent skip reasons. Per-example generation/JSON dumps are
off by default; warnings/errors are not hidden. `--verbose-runtime` adds detail.
Percentages average per-example reasoning masks in the current batch, excluding
question, answer and chat-template tokens. They are not cumulative averages.

Detailed files: `run_manifest.json`, `invalid_rows.jsonl`, `metrics.jsonl`,
`windows.jsonl`, `controls.jsonl`, `samples_rank_N.jsonl`. Window logs include
Beta statistics, reward components, all loss observations and signed/clipped
changes, virtual losses/norms/LR/resolution, raw/scaled delta and RMS, D, old/new
weights, actor advantage/norm/same-state Beta KL, independent valid counters,
alias/branching/mask diversity, rendered context lengths, KL(FULL||MASK), forward
counts, scored/generated tokens, cache bytes and rank-max timings.

Atomic checkpoints every 25 steps and the endpoint include adapter/tokenizer,
`training_state.pt`, and `checkpoint_manifest.json`. Persist phase/warmup,
optimizer/scheduler, latent/effective weights, pending state/action/old weights,
controls/labels, partial loss/grounded observations, both adaptive channels,
sampler cursors and all independent RNG states. Resume is supported after an
optimizer step inside a window, not inside an uncommitted microbatch.
Redesign rejects old-method checkpoints and changed reward/template/phase/answer
definitions; start a new run. Changing the total horizon also requires a new run.
Changing output/scratch path or verbosity is allowed. Never overwrite checkpoints.

Training explicitly shuts down the vLLM engine and collects its objects before
destroying vLLM model-parallel groups and the training process group. No wake-up
or extra model allocation is needed during cleanup. Rank-tagged shutdown stages
and Python faulthandler output distinguish engine/group/interpreter teardown
failures. This corrects the missing explicit cleanup path, but is not proof that
every server-side native SIGABRT/SIGSEGV is fixed; CUDA/vLLM needs a real rerun.

## Run

Use the existing training environment with torch, transformers, peft, vLLM,
PyYAML, numpy, tqdm and the repository's evaluation dependencies.

### Parameterized Experiment Script

`scripts/train_adaptive_graph_ftrl.sh` provides an environment-variable interface without
editing YAML or changing defaults of the older entrypoints:

```bash
# A fresh output directory is generated when RUN_DIR is unset.
unset RUN_DIR
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh

# Example alternative: no FULL warmup, 32 duel controls from a pool of 256.
DUEL_PROBLEMS=32 HELDOUT_PROBLEMS=256 WARMUP_STEPS=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 MAX_STEPS=75 EVAL_STEPS=25,50,75 \
EVAL_DATASETS=aime24,aime25 \
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh
```

Defaults: `DUEL_PROBLEMS=8`, `HELDOUT_PROBLEMS=128`, `WARMUP_STEPS=5`,
`MIN_VALID_CONTROLS=8`, `MAX_STEPS=75`, `SAVE_EVERY=25`, hard teacher selection,
MASK with final answer, GPUs 0/1/2/3 and train/eval seeds 42. `DUEL_PROBLEMS`
maps to `control_batch_size`: heldout questions used to score virtual branches,
not the main batch for FULL/MASK gradients (still 32 by default). The selected
controls are frozen for a window, with per-step duels in hard mode. If reducing
duel controls below 8, also reduce `MIN_VALID_CONTROLS` explicitly.

When unset, `EVAL_STEPS` includes every scheduled checkpoint plus the final step
(25/50/75 for 75 steps, 25/50/75/100 for 100 steps). `EVAL_DATASETS` defaults to
`aime24,aime25`; AIME26 is also supported. Impossible checkpoint schedules and
invalid control counts fail before launching training. Extra arguments are
passed to the existing training CLI; use the environment options for settings
managed by this script so its schedule checks match the executed run.

`RUN_ROOT` defaults to `$HOME/dsd/outputs`; `RUN_DIR` overrides the generated
unique output path. Existing run manifests/metrics/logs are refused to avoid
overwriting experiments. The combined log is `$RUN_DIR/train_eval.log`.
Generated names include variant/teacher mode, rho, RMS decay/floor, delta clip,
duel/pool/min-valid counts, FULL warmup, window size, actor warmup/LR, exposure
temperature, reward coefficients, main batch/LR, answer visibility, train/eval
seeds and total steps. A 12-character config hash, UTC timestamp and process ID
distinguish runs. Other settings are represented by the hash, not long filename
fragments; unusually long names are capped to fit filesystem filename limits.
**Every resolved training setting** is saved in `$RUN_DIR/config.yaml`.
`experiment.json` records source config, hash and evaluation settings. YAML is
resolved and validated before GPU launch; the saved YAML is used for training.

`EVAL_WORK_DIR` defaults to a short, unique sibling `.runtime/run-*` directory;
the long descriptive run name is deliberately excluded from temporary paths to
avoid Unix socket path limits. All temporary, Triton,
TorchInductor, vLLM and UV cache paths are exported there before launch. No cache
or checkpoint cleanup or compiler probe is performed. Choose writable storage
with enough space; changing a path does not reserve disk capacity.

`TRAIN_SEED`, `EVAL_SEED`, `EVAL_GPU_MEMORY_UTILIZATION`, `CONFIG`,
`DATASET_JSONL` and `PYTHON_BIN` are configurable. The new `--control-pool-size`
and `--eval-seed` training flags are also available directly. The wrapper
delegates to `run_train_then_eval.sh`, preserving completion checks, failure
status and the graph-based evaluation recipe and result locations. It does not
change the training method or model. Full option list:

```bash
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh --help
```

### Hard Hyperparameters

The script's default config is `configs/hard_experiment.yaml`. Override optional
settings with environment variables or equivalent training CLI flags. Precedence
is CLI > explicitly set environment override > YAML > dataclass default. Unset
variables preserve custom YAML values. Hard mode still duels every adaptive step;
`WINDOW_STEPS` changes control reuse and alpha sampling, not this duel frequency.

| Environment | CLI | Default |
| --- | --- | --- |
| `FTRL_RHO` | `--ftrl-rho` | `0.9` |
| `DELTA_RMS_DECAY` | `--delta-rms-decay` | `0.9` |
| `DELTA_RMS_FLOOR` | `--delta-rms-floor` | `1e-4` |
| `DELTA_SCALED_CLIP` | `--delta-scaled-clip` | `1.0` |
| `WINDOW_STEPS` | `--window-steps` | `5` |
| `ACTOR_WARMUP_WINDOWS` | `--warmup-windows` | `3` |
| `CONTROLLER_LR` | `--controller-lr` | `0.01` |
| `EXPOSURE_TEMPERATURE` | `--exposure-temperature` | `0.05` |
| `LAMBDA_GT` | `--lambda-gt` | `0.01` |
| `GAMMA_DISCOUNT` | `--gamma-discount` | `0.95` |
| `LEARNING_RATE` | `--learning-rate` | `5e-6` |

`D = rho * D + clip(delta / historical_RMS, -delta_scaled_clip, delta_scaled_clip)`.
The default clip reproduces the previous fixed `[-1,1]` behavior. Invalid duels
still hold D, weights and RMS statistics without decay. Positive D selects MASK;
zero/negative D selects FULL, for the next optimizer step. `FTRL_ETA` /
`--ftrl-eta` and `FTRL_MIN_WEIGHT` / `--ftrl-min-weight` affect **soft mode only**.
CLI also exposes normalizer decay/floors/clip, controller gradient clip, Adam
betas/epsilon, weight decay, student gradient clip, answer temperature and graph
pair cap. All retain their previous defaults. Old checkpoints without the new
delta clip field resume only with its original default `1.0`.

```bash
unset RUN_DIR CONFIG
CUDA_VISIBLE_DEVICES=0,1,2,3 \
FTRL_RHO=0.8 DELTA_RMS_DECAY=0.9 DELTA_RMS_FLOOR=1e-4 DELTA_SCALED_CLIP=1 \
DUEL_PROBLEMS=8 HELDOUT_PROBLEMS=128 MIN_VALID_CONTROLS=8 WARMUP_STEPS=5 \
WINDOW_STEPS=5 ACTOR_WARMUP_WINDOWS=3 CONTROLLER_LR=0.01 \
EXPOSURE_TEMPERATURE=0.05 LAMBDA_GT=0.01 GAMMA_DISCOUNT=0.95 \
MAX_STEPS=75 TRAIN_SEED=42 EVAL_SEED=42 \
EVAL_STEPS=25,50,75 EVAL_DATASETS=aime24,aime25 \
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh

# Equivalent rho override, without a separate YAML:
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh --ftrl-rho=0.8
```

### Existing Entrypoints

```bash
cd ~/dropout-self-distillation
source dsd/bin/activate
tmux new -s graph_redesign
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_STEPS=50 EVAL_STEPS=25,50 EVAL_DATASETS=aime25 \
RUN_DIR=$HOME/dsd/outputs/graph_redesign_eta3_step50 \
bash adaptive_graph_ftrl_redesign/scripts/run_train_then_eval.sh \
  2>&1 | tee graph_redesign_step50.log
```

The shell script default is 100 steps with checkpoints 25/50/75/100. It verifies
fresh completion records from every rank and the final checkpoint before eval.
A checkpoint left by an older launch cannot authorize evaluation. If the launcher
fails after verified completion (for example, a native teardown crash), it warns,
evaluates saved checkpoints, then preserves the nonzero training exit code.
An incomplete/failed training run or broken final adapter never triggers eval.
Independent evaluation always uses an explicit nonempty path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m adaptive_graph_ftrl_redesign.evaluate \
  --run-dir "$HOME/dsd/outputs/graph_redesign_eta3_step50" \
  --steps 25,50 --datasets aime25 --gpus 0,1,2,3 --seed 42
```

For hard teacher selection on GPUs 4/5/6/7, with a duel at **every one of the
50 steps**, explicitly disable FULL warmup. Use a new output directory, not a
soft-run checkpoint. Exposure alpha and actor updates still use H=5:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=4,5,6,7 \
CONFIG=adaptive_graph_ftrl_redesign/configs/graph_hard_answer.yaml \
RUN_DIR=$HOME/dsd/outputs/graph_hard_ftl_every_step_50 \
MAX_STEPS=50 TRAIN_SEED=42 EVAL_SEED=42 \
EVAL_STEPS=25,50 EVAL_DATASETS=aime25 EVAL_GPU_MEMORY_UTILIZATION=0.9 \
bash adaptive_graph_ftrl_redesign/scripts/run_train_then_eval.sh \
  --full-opsd-warmup-steps 0 \
  2>&1 | tee graph_hard_ftl_every_step_50.log
```

Omit the warmup override to keep five initial FULL-only steps and 45 per-step
adaptive duels. Eta does not affect hard argmax selection. To evaluate saved
checkpoints independently (including after a training teardown error):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m adaptive_graph_ftrl_redesign.evaluate \
  --run-dir "$HOME/dsd/outputs/graph_hard_ftl_every_step_50" \
  --steps 25,50 --datasets aime25 --gpus 4,5,6,7 --seed 42 \
  --gpu-memory-utilization 0.9
```

Evaluation remains question-only, thinking enabled, temperature 1.0, top-p .95,
top-k -1, 12 samples, 38912 new tokens, **Avg@12, not Pass@12**. No graph,
reference, answer label or controller at inference. No test-answer tuning or
automatic best-checkpoint selection; report fixed endpoints unless specifying
a separate peak-selection protocol.

Redesign evaluation reuses the same launcher as SE-OPSD:
`graph_based_scripts/run_graph_based_eval_opsd_4gpu.sh` ->
`graph_based_scripts/run_graph_based_eval.sh` -> `uv run python eval/evaluate_math.py`.
Install `uv` and use the working SE evaluation environment, selected by
`UV_PROJECT_ENVIRONMENT` (default `dsd`). Model, checkpoint, sampling settings,
seed, GPU list, memory budget and output file are forwarded explicitly. Training
still uses the active Python environment. `--launcher python` opts into the
active-Python evaluation path. Reusing the launcher does not repair missing compiler headers or CUDA
libraries in the server environment; check GCC stderr if compilation still fails.

### Configurable AIME24/AIME25/AIME26 Script

For checkpoints 25/50 on GPUs 0/1/2/3, use a direct wrapper of the existing
`graph_based_scripts/run_graph_based_eval_opsd_4gpu.sh` (which calls the same
`run_graph_based_eval.sh` and `uv run python eval/evaluate_math.py`). There is no
separate inference implementation and no active-Python fallback:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_DIR="$HOME/dsd/outputs/graph_hard_ftl_every_step_50" \
EVAL_WORK_DIR="$HOME/dsd/eval_runtime" \
EVAL_DATASET=aime25 EVAL_STEPS=25,50 \
bash adaptive_graph_ftrl_redesign/scripts/run_eval.sh
```

Use `EVAL_DATASET=aime24` for AIME24, and `EVAL_STEPS=50` to evaluate only
checkpoint 50. Use `EVAL_DATASET=aime26` for AIME26, loaded from
[`MathArena/aime_2026`](https://huggingface.co/datasets/MathArena/aime_2026)
(train split, `problem`, integer `answer`, `problem_idx`). Defaults are unchanged.
No source edits are needed. Run `bash
adaptive_graph_ftrl_redesign/scripts/run_eval.sh --help` for all options.
The old `run_eval_aime25_25_50.sh` name delegates to this same configurable script.
Supported overrides include `CUDA_VISIBLE_DEVICES` (four GPUs), `EVAL_SEED`,
`VAL_N`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`,
`MAX_NEW_TOKENS`, `MAX_MODEL_LEN`, `ENABLE_THINKING` (1 or 0),
`EVAL_GPU_MEMORY_UTILIZATION`, `MODEL_PATH`, `RESULTS_DIR`, `EVAL_WORK_DIR`,
and the preflight limits below. Defaults preserve the graph-based OPSD recipe.
Run datasets sequentially when sharing the same four GPUs.

The supplied disk-full log instead evaluated `graph_redesign_eta1_step50`.
Set `RUN_DIR="$HOME/dsd/outputs/graph_redesign_eta1_step50"` for that run.
The script requires an explicit existing run path and validates all requested
adapters before inference, so a missing checkpoint fails immediately.

The logged root error was GCC writing `/tmp/cc*.s`: **No space left on device**,
followed by Triton failure and EngineDeadError after 348/360 completions. Moving
only Python's generated source does not move GCC's intermediate assembly.
The wrapper exports `TMPDIR`, `TMP`, `TEMP`, `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR` and `VLLM_CACHE_ROOT` under `EVAL_WORK_DIR` **before uv
or workers start**. GCC uses `TMPDIR` for temporary files, as documented in
[GCC environment variables](https://gcc.gnu.org/onlinedocs/gcc/Environment-Variables.html).
It does not change the Hugging Face model cache or delete anything automatically.
The default work directory is `$RUN_DIR/eval_runtime`, not system `/tmp`.
Select another filesystem with `EVAL_WORK_DIR` if the run filesystem is small.

Preflight checks writable storage, available bytes/inodes, readable LoRA adapters
and four distinct GPU IDs, then starts evaluation. It does not compile/load a
test library or create/remove `gcc-probe-*` directories. No cache or checkpoint
cleanup is performed. If storage is insufficient, select another filesystem
with `EVAL_WORK_DIR` (temporary/cache files) and `RESULTS_DIR` (results/logs).
The checks run before each checkpoint. The operational safety floor is 10 GiB / 10,000 available
inodes per checked filesystem (`EVAL_MIN_FREE_GIB` / `EVAL_MIN_FREE_INODES`);
this is not a guaranteed upper bound on disk use or a reservation against other
jobs filling the disk later. Choose a sufficiently large writable filesystem.
Use `EVAL_PREFLIGHT_ONLY=1` to check without loading vLLM or downloading a model.

Defaults remain temperature 1.0, top-p .95, top-k -1, min-p/presence penalty 0,
12 samples, thinking, 38912 completion tokens, context 40960, seed 42,
TP=4 and VRAM utilization .9. The working graph-based UV environment is still
`UV_PROJECT_ENVIRONMENT` (default `dsd`). Results and logs are separate per
checkpoint in `$RUN_DIR/eval_<dataset>/` (`RESULTS_DIR` overrides this). Pipeline
failures propagate through `tee`; a failed checkpoint never proceeds to the next.
Preflight does not verify CUDA performance, available GPU VRAM or AIME accuracy.

Evaluation inherits `CUDA_VISIBLE_DEVICES` unless `--gpus` is provided. Its
vLLM memory budget is separate from training and defaults to 0.9. On shared
GPUs, set `--gpu-memory-utilization 0.4` for standalone evaluation or
`EVAL_GPU_MEMORY_UTILIZATION=0.4` for the train-then-eval script. Check free
memory on every selected GPU first. A smaller budget may still be insufficient
for the unchanged 40960-token context; no automatic token-limit reduction is
performed. A requested checkpoint that cannot create its LoRA request fails
instead of silently evaluating the base model.

Choose a config with `CONFIG=adaptive_graph_ftrl_redesign/configs/NAME.yaml` and
a **fresh RUN_DIR** for every variant:

| Config | Comparison |
| --- | --- |
| `graph_hard_answer.yaml` | Hard FULL/MASK leader, expert duel every optimizer step, MASK has answer |
| `full.yaml` | FULL-only baseline, same data partition and student recipe |
| `prefix_answer.yaml` | Adaptive prefix with same reward/control/FTRL |
| `graph_answer.yaml` | Primary adaptive graph with explicit answer |
| `frozen_beta.yaml` | Beta(2,2) fixed actor, graph and learned FTRL |
| `fixed_mixture.yaml` | Adaptive actor with fixed .5/.5 mixture; duel still measured |
| `graph_no_answer.yaml` | Answer-visibility ablation |
| `no_grounding.yaml` | lambda_gt=0; retain grounded-scoring compute |

Graph-vs-Prefix must match answer visibility, reward/scoring spans, FULL warmup,
control objective and recipe. Learned-vs-frozen and fixed-vs-FTRL comparisons are
needed to test adaptation/competition. Extra MASK/reference/control/replay
compute is logged; equal student steps do not mean equal GPU cost.

Resume example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m adaptive_graph_ftrl_redesign.train \
  --config adaptive_graph_ftrl_redesign/configs/graph_answer.yaml \
  --total-updates 50 --output-dir "$HOME/dsd/outputs/graph_redesign_eta3_step50" \
  --resume "$HOME/dsd/outputs/graph_redesign_eta3_step50/checkpoint-25"
```

## Offline Checks

```bash
python -m pytest -q
python -m adaptive_graph_ftrl_redesign.smoke_test
python -m adaptive_graph_ftrl_redesign.smoke_test \
  --teacher-selection hard
python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  -m adaptive_graph_ftrl_redesign.tests.distributed_check \
  --output-dir /tmp/graph_redesign_4rank_smoke
```

Add `--teacher-selection hard` and a fresh `--output-dir` to the four-rank check
to exercise per-step hard selection and pending-window resume.

Tiny tests instantiate local Qwen3/LoRA without downloading pretrained weights.
The four-rank test mocks the inference engine; it is not a real CUDA/vLLM run.
See `VALIDATION.md` for actual checks and unverified GPU/accuracy behavior.
