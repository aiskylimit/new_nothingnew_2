# Validation

## Self-Contained Branch Cleanup, 2026-10-10

- Removed 185 unrelated tracked files from `adaptive_graph_vs_opsd`, including
  legacy adaptive/FULL, graph-based/free and SE trainers, experiment registries,
  root ablations, obsolete launchers and their tests. Other branches are unchanged.
- Moved only required data/parser, graph closure, policy, model, runtime, RNG,
  sampler, rollout, logging and evaluation helpers into the redesign. AST checks
  against the pre-cleanup commit confirm identical migrated runtime methods,
  policy, parser, model loading, RNG and rollout bodies. Dependency versions and
  evaluation recipe/result paths are unchanged; no legacy package import remains.
- Migrated parser/mask and rollout regressions and made tiny fixtures independent.
  `python -m pytest -q`: **103 passed** in the clean pruned worktree, without a
  legacy PYTHONPATH overlay. Tests explicitly block imports of removed packages.
- Hard and soft tiny-model smoke tests pass, including actor/expert updates and
  bit-exact mid-window resume. Four-rank CPU/Gloo checks pass for both hard and
  soft, with mocked vLLM, effective batch 32, synchronized weights, checkpoint
  resume, completion proofs and ordered teardown. All six shell scripts pass
  syntax checks; AIME24/AIME25 checkpoint 25/50/75 eval dry-run passes.
- Tests use the existing local Python 3.14 CPU environment, not the locked server
  Python 3.13 GPU environment. No real CUDA/vLLM inference, pretrained-model
  training or AIME accuracy evaluation was run. `uv` is not installed locally;
  the retained evaluation shell recipe is exercised through its mocked launcher.
- Earlier sections below describe historical validation before this cleanup.

## Configurable Hard Hyperparameters, 2026-10-09

- Training CLI exposes FTRL rho/eta/floor, delta RMS decay/floor, symmetric delta
  clip, controller/window/warmup settings, normalizer settings and Adam/LR knobs.
  Script environment overrides cover the listed hard-mode experiment settings.
  Defaults preserve the previous formulas, including delta clip 1.0. Hard mode
  still refreshes every adaptive step; eta/min-weight remain soft-only.
- Default `configs/hard_experiment.yaml` lists the hard experiment parameters.
  A lightweight shared parser resolves and validates all overrides before GPU
  launch. The launcher saves a full `config.yaml` and evaluation metadata, with
  descriptive parameter/seed names plus a config hash, UTC timestamp and PID.
  Unlisted parameters change the hash; overly long filenames are capped.
  Runtime/cache directories have short independent names to avoid propagating
  the descriptive output name into Unix socket paths. No cleanup is performed.
- Combined redesign + legacy adaptive + FULL suites: **188 passed**, one existing
  Python 3.14/TorchScript warning. Tests cover env/CLI/YAML precedence, resolved
  snapshots, auto-naming, hashed unlisted settings, invalid values before launch,
  generated runtime paths, configurable rho/RMS/clip math, and old-checkpoint
  acceptance only with the original clip. Shell syntax, Ruff F and diff checks pass.
- Verification used the tracked legacy dependency exported outside the worktree
  via PYTHONPATH, as in the previous validation. Its existing local deletion was
  not changed. No real GPU/vLLM training or AIME evaluation was run.

## Parameterized Experiment Launcher, 2026-10-09

- Added `scripts/train_adaptive_graph_ftrl.sh`: defaults to 8 duel-scoring controls from
  a heldout pool of 128, FULL warmup 5, hard selection, answer-visible MASK,
  75 training steps, checkpoint/eval 25/50/75, AIME24/AIME25 and both seeds 42.
  Existing config/entrypoint defaults are unchanged. Training CLI now exposes
  `--control-pool-size` and `--eval-seed`.
- Fifteen focused tests pass: actual shell delegation with mocked Python
  train/completion/eval calls, real parser/config validation, default/custom
  values, all runtime/cache paths, seed forwarding, generated eval schedules,
  invalid parameter rejection, existing-log protection and failure status.
  No existing cache file is removed. Shell syntax, Ruff F and whitespace checks pass.
- Combined redesign + legacy adaptive + FULL suites: 175 passed, one existing
  Python 3.14/TorchScript warning. The legacy `adaptive_graph_ftrl_opsd` directory
  was deleted from the local worktree during this task; its tracked HEAD version
  was exported to an external temporary directory and supplied via PYTHONPATH
  for verification. The worktree deletion was not restored or included in this
  change. Without that dependency, the local trainer cannot import normally.
- No GPU/vLLM experiment or benchmark evaluation was launched. Shell mocks test
  configuration and orchestration, not real GPU memory, throughput or accuracy.

## AIME26 Evaluation Support, 2026-10-08

- Combined suites: 160 passed, one existing Python 3.14/TorchScript warning.
  Focused evaluation/script tests: 31 passed. Shell syntax and Ruff F checks passed.
- Added `aime26` to the script and shared evaluator CLI; loads
  `MathArena/aime_2026`, train split, using `problem`, integer `answer` converted
  to a label string and `problem_idx`. Schema checked against the publisher's
  dataset card. Offline tests exercise real evaluator control flow with mocked
  dataset/inference, including integer labels, question-only prompts and score.
- AIME24/AIME25 defaults are unchanged. Script forwarding, output/log naming
  and checkpoint selection tested for AIME26. No real GPU/AIME26 run performed.

## Remove Compiler Probe From Evaluation, 2026-10-08

- Combined suites: 157 passed, one existing Python 3.14/TorchScript warning.
  Focused script suite: 10 passed, including preservation of existing cache
  files and successful preflight with CC pointing to a nonexistent compiler.
- Removed the preflight compiler/shared-library probe entirely, including its
  temporary directory cleanup. The supplied server traceback failed during
  `gcc-probe-*` cleanup, after storage and adapter validation had succeeded.
- Storage/checkpoint checks and configured temporary/cache/result directories
  remain; no cache/checkpoint cleanup is performed. The actual evaluator still
  requires its usual compiler/CUDA environment. No real GPU evaluation run.

## Configurable AIME24/AIME25 Evaluation, 2026-10-08

- Combined redesign + legacy adaptive + FULL suites: 157 passed, one existing
  TorchScript/Python 3.14 warning. Focused script suite: 10 passed.
- Generic `scripts/run_eval.sh` calls the same graph-based evaluator; the old
  AIME25 script delegates to it. Tested AIME24 with checkpoint 50 only and
  overrides for sampling, seed, thinking, context length and VRAM utilization.
  Output/log paths contain the selected dataset. Invalid dataset fails before
  inference, and help works without a checkpoint or UV installation.
- Existing real storage/GCC preflight tests and failure propagation remain green.
  Temporary/compiler caches default to the run filesystem, not system `/tmp`.
  Shell syntax, Ruff F checks and git whitespace checks passed.
- No real CUDA/vLLM evaluation was run. Capacity checks cannot reserve disk
  against other processes, and do not guarantee freedom from GPU/native faults.

## Standalone Graph-Based AIME25 Script, 2026-10-08

- Combined redesign + legacy adaptive + FULL suites: 154 passed, one existing
  TorchScript/Python 3.14 warning. Seven new tests exercise actual graph-based
  shell scripts with only the final inference command replaced, plus real
  storage checks and GCC compile/shared-library load in the configured TMPDIR.
- Confirmed checkpoint 25/50, GPU 0/1/2/3, UV environment, sampling settings,
  thinking, context, seed, VRAM budget and all compiler/cache paths are forwarded.
  Missing/broken checkpoint, low bytes/inodes, preflight-only and inference
  failure cases are tested; tee preserves nonzero inference status.
- Supplied server log explicitly reports GCC `No space left on device` writing
  `/tmp/cc*.s`, before Triton/EngineDeadError at 348/360 completions. The new
  standalone script redirects both GCC/Python temporary storage and compilation
  caches before launching uv/vLLM. No sampling or inference implementation changes.
- No server filesystem access, real GPU/vLLM inference or AIME results verified.
  Free-space checks do not reserve capacity or prevent other jobs filling a disk.

## Training Teardown And Evaluation, 2026-10-08

- Combined redesign + legacy adaptive + FULL suites: 147 passed, one existing
  TorchScript/Python 3.14 warning. Fourteen new tests cover ordered teardown,
  vLLM V1 core/V0 executor shutdown, idempotent close, fresh completion proofs,
  actual shell handoff to eval, stale/incomplete/corrupt checkpoint rejection,
  preservation of training failure status, and propagation of eval failures.
- Four CPU/Gloo ranks with mocked vLLM passed in hard and soft modes, including exact
  checkpoint/resume, per-rank completion records, final adapter validation,
  engine shutdown before training group destruction and clean launcher exit.
- Compared the `graph_based` train-then-eval shell path and its inherited
  `OPSDTrainer` external-launcher engine construction. Its shell also uses
  `set -e`; copying it does not bypass nonzero training exit status. The redesign
  previously destroyed the training group with a live vLLM engine and no
  explicit engine shutdown. Eval already uses the graph-based/SE launcher.
- Checked shutdown entry points against upstream vLLM 0.11.0 source. Native
  SIGABRT/SIGSEGV cannot be attributed conclusively from the supplied elastic
  launcher traceback alone. Ordered cleanup and faulthandler diagnostics are
  implemented, not claimed to prove a CUDA-native fix without a real rerun.
- No method, teacher weights, optimizer, sampling defaults or seed changes.

## Hard Teacher Update, 2026-10-08

- Combined redesign + legacy adaptive + FULL suites: 133 passed; one existing
  PyTorch TorchScript/Python 3.14 deprecation warning.
- Hard-mode offline smoke passed with a tiny actual Qwen3/LoRA: graph masks,
  selected-expert student update, per-step exact virtual AdamW duel, actor update
  after warmup and bit-exact pending-window checkpoint/resume.
- Four CPU/Gloo ranks with mocked vLLM passed: batch 32 / accumulation 8,
  per-step full-batch pure FULL/MASK gradients, binary expert weights, expert
  refresh between steps, window-end-only replay, synchronized state and exact
  mid-window resume.
- Tests verify exact selected teacher targets/actual gradients without mixing,
  no same-step use of next weights, no double expert update at window end,
  FULL tie-breaking, invalid-duel state freeze, realized MASK frequency for the
  actor, soft-default preservation and old soft checkpoint compatibility.
- No real GPU/vLLM run or AIME evaluation was performed for this change.

## Initial Redesign

2026-10-07, local CPU environment, Python 3.14, PyTorch/Transformers/PEFT installed.
No CUDA or real vLLM inference available; no Qwen3-4B download or full experiment
was started. Implementation is not a claim of improved benchmark accuracy.

- Combined redesign + legacy adaptive + FULL unit suites: 100 passed.
  Tiny actual Qwen3/LoRA, not a mocked training loss.
- Offline end-to-end smoke passed: answer split, graph sampling, FULL warmup,
  phase transition, raw mixture, causal reference scoring, frozen target replay,
  virtual optimizer competition, FTRL and actor update after three valid rewards.
- Bit-exact resume passed from warmup and an adaptive pending window, including
  pending action, loss observations, selected control inputs, all sampling streams,
  final LoRA parameters and controller feedback.
- Four Gloo ranks with mocked vLLM passed: accumulation 8 / batch 32; exactly 32
  main forwards and 64 pure-expert gradient evaluations at refresh, all 32 last
  rollouts replayed, 4 reference teacher forwards, 3 evaluations of 8 direct
  controls, no control generation, synchronized weights and exact mid-window resume.
- Virtual AdamW tests match independent real optimizer steps, both with initial
  empty state and existing momentum, LR schedule, weight decay and clipping.
  Check live parameters, optimizer/scheduler, .grad, modes, buffers and RNG restore.
- Independent invalid channels, historical RMS without centering, old-weight
  actor attenuation, independent warmup counters, frozen/fixed ablations and
  label offsets/truncation tested. Existing parser/mask/raw-KL tests remain green.
- Validate-only on 500 lines of the available local
  `graph_based_outputs/hypergraphs_train_hf.jsonl` passed: 478 eligible, 22 excluded;
  used cached AIME question fingerprints in offline mode, without loading a model.
  This is not validation of the server's operations_v3 file, which is not local.
- CLI help, Boolean flags, evaluation dry-run, shell syntax, Ruff F checks and
  whitespace checks passed. Eval dry-run preserves thinking Avg@12, temp 1.0,
  top-p .95, top-k -1 and seed 42 for checkpoints 25/50 on AIME25.

Unverified: GPU peak memory, actual colocated vLLM behavior and native teardown,
server dataset contents, long-reference throughput/disk bandwidth, direct-answer
surrogate reliability and AIME accuracy. The real-model training recipe defaults
to vLLM; the engine mock proves orchestration, not CUDA performance.

Scope: new package files only, plus a controller factory extension in the old
runtime to reuse initialization without duplicating or changing its default method.
