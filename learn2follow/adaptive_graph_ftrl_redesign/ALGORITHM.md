# Adaptive Graph vs FULL OPSD

This document describes the implemented algorithm on branch
`adaptive_graph_vs_opsd`, not an unimplemented proposal or a claim of superior
benchmark accuracy. The main experimental setting is **graph masking with hard
teacher selection**. Soft aggregation and FULL-only training are explicit
alternatives in the same package.

## 1. Components And Information Flow

There are three adaptive components:

1. A dependency graph limits which reasoning steps can be removed together.
2. A ten-parameter Beta policy chooses a retained-reasoning exposure per window.
3. A discounted hard leader chooses FULL or MASK for each student optimizer step,
   using an exact one-step virtual AdamW comparison on heldout answer NLL.

```text
reference solution -> answer split -> reasoning graph -> candidate masks
                                                         |
lagged state -> Beta policy -> alpha -> sampled mask ------+
                                       |
question + full reference ------------> FULL teacher -> q_F
question + retained reference --------> MASK teacher -> q_M
question ----------------------------> student rollout y
                                       |
                         FULL/MASK score that same y
                                       |
                     current leader selects one KL target
                                       |
                          actual student AdamW update

pure FULL/MASK gradients -> virtual AdamW branches -> heldout answer NLL
                                                -> delta -> next leader

window loss observations + MASK reference log-probabilities
                                                -> actor reward -> next policy
```

FULL and MASK are context-conditioned roles of the current self-teacher, not
two independently trained backbone models. Teacher scoring is detached and uses
the same pre-update model snapshot for both contexts. The trained student only
requires the question at inference time.

## 2. Data And Answer Separation

Each eligible example contains a question, original reference solution,
reasoning nodes with source spans, dependency incidences, a terminal answer
block and a gold answer label.

The parser separates the terminal answer **before** creating mask candidates.
Explicit answer spans/blocks take precedence; otherwise terminal markers and
balanced-brace boxed expressions are used. Nested LaTeX braces are supported.
An intermediate boxed expression is not automatically the final answer.

If a reasoning node contains derivation followed by the answer, retain its
reasoning prefix. Remove answer-only nodes, while preserving the IDs of retained
nodes. Reject suspicious outgoing dependencies from removed nodes, cyclic graphs,
ambiguous spans and examples without reasoning nodes. Invalid records are counted
and written to `invalid_rows.jsonl`; MASK never silently receives the raw
reference as an invalid-example fallback.

FULL uses the original reference text, so answer separation does not delete
content from FULL. MASK concatenates retained source-order reasoning fragments.
With `mask_teacher_include_final_answer=true`, append the separated answer with
an explicit `Final answer:` heading when it does not already have one. With the
flag false, omit that explicit block. This does **not** scrub intermediate
calculations that might reveal the answer.

The labeled control pool is disjoint from the main optimizer data. Configured
benchmark questions are excluded by question fingerprints before splitting.
The default benchmark exclusion list is AIME24/AIME25. Any additional benchmark
must also be added to `benchmark_tasks` for training-time exclusion; merely
selecting it for later evaluation does not extend that list.

For comparisons, FULL-only and Graph use the same graph-eligible data filtering
and reserve the same heldout pool, even though FULL-only does not use adaptive
feedback. This FULL-only baseline is a recipe-matched comparator, not necessarily
identical to every older OPSD trainer in the repository.

## 3. Graph Masks And Exposure

Let `V_i` be reasoning nodes, `l_v` their cached token lengths, and `T` the
removed node set. A mask must be descendant-closed:

```math
v\in T \Longrightarrow \operatorname{Desc}(v)\subseteq T.
```

Removing a prerequisite therefore also removes its recorded dependents. No
last-step protection or balancedness constraint is added after answer separation.

The candidate pool contains every single-seed descendant closure, sampled or
enumerated unions of two seed closures, and the all-hidden mask `T=V_i`.
Deduplicate by node set and remove the empty mask. This is a finite intervention
pool, not an enumeration of all descendant-closed subsets. `max_pair_candidates`
defaults to 4096; the pair reservoir uses an example-specific stable seed.
Setting the cap to zero enumerates all pairs, not zero pairs.

Define retained-token exposure:

```math
L_i=\sum_{v\in V_i}l_v,\qquad
m_i(T)=\sum_{v\notin T}l_v,\qquad
e_i(T)=m_i(T)/L_i.
```

Question, answer and chat-template tokens are excluded. Zero-token reasoning
nodes are rejected by the redesign. Rendered context lengths are logged
separately because concatenation tokenization need not equal summed node costs.

Group candidates by integer retained count `m`, then sample:

```math
P(m\mid\alpha,G_i)=
\frac{\exp(-|m/L_i-\alpha|/\tau)}
{\sum_{m'}\exp(-|m'/L_i-\alpha|/\tau)}.
```

After choosing a level, uniformly sample a mask in that group. A level with many
masks does not gain additional prior mass. Only the selected MASK is teacher
scored. All-hidden produces answer-only reference context when answer visibility
is true, or no reference context when false; the question/template remains.

Per-step `mask_token_percent` is the mean of `100*(1-e_i)` over main examples.
`mask_step_percent` is the mean of `100*|T_i|/|V_i|`. These are current-batch
reasoning-mask statistics, not cumulative percentages or total prompt savings.

## 4. Exposure Policy And Window Timing

Let `H=window_steps` and `K=ceil(total_updates/H)`. The scheduled window index `k`
includes FULL warmup windows. The lagged state is:

```math
s_k=[\min(1,k/\max(K-1,1)),\ Z_{\delta,\mathrm{last}},\
w_{M,\mathrm{start}},\ Z_{R,\mathrm{last}}].
```

Only completed valid observations enter the last-feedback fields. Hard mode's
`w_M,start` is binary. Its leader may subsequently change within the window,
but the saved state and sampled action do not change.

```math
(a_k,b_k)=1+\operatorname{softplus}(Ws_k+c),\qquad
W\in\mathbb R^{2\times4},\quad c\in\mathbb R^2.
```

There are exactly ten trainable policy parameters. Initialization is `W=0`,
`c=log(expm1(1))`, giving Beta(2,2). Sample a detached scalar alpha once per
window; do not differentiate through that sample or graph sampling. Alpha is a
target for **retained** exposure, not the masked fraction. Each example still
samples its own candidate according to its feasible exposure levels.

Default FULL warmup is five optimizer steps, followed by adaptive windows of
five steps. The optimizer and LR schedule do not reset at this transition.
FULL warmup has no alpha, MASK scoring, control evaluation, virtual branches,
grounded scoring or adaptive-statistic updates. Hard mode starts with FULL and
`D=0`; the first adaptive step also uses FULL until a duel selects the next leader.

## 5. Student Distillation Objective

For each main question, generate one student continuation `y_i` from a
question-only, non-thinking prompt. FULL and MASK score this same continuation
using their respective privileged thinking contexts.

With distillation temperature `T_d`, obtain detached distributions `q_F` and
`q_M`, and student distribution `p_theta`, using FP32 softmax/log-softmax. For
expert `j`, the objective is:

```math
L_j(\theta)=\frac1B\sum_{i=1}^B\frac1{|y_i|}
\sum_t\mathrm{KL}(\operatorname{stopgrad}(q_{j,i,t})\|p_{\theta,i,t}).
```

Only generated continuation tokens are scored, with the causal shift. Average
over tokens per example, then examples, not over a flattened variable-length
batch. Do not train on the privileged reference prefix.

Hard mode chooses one expert for the **entire optimizer step**, including all
accumulation microbatches. Its target and actual gradient are exactly that
expert's target/gradient, with weights `[1,0]` or `[0,1]`; there is no mixture.
Both experts are nevertheless scored, and their pure gradients computed, to
allow the unselected teacher to compete in the next duel.

Soft mode instead uses `q=w_F*q_F+w_M*q_M` in probability space. Its raw gradient
obeys `g_mix=w_F*g_F+w_M*g_M`, before gradient clipping and AdamW. Mixing logits
would not implement this objective. Soft weights stay fixed for a whole window.

## 6. Direct-Answer Controls, Not Generated Probes

Default script settings reserve 128 labeled questions and select eight controls
per adaptive window. Freeze their tokenized inputs, labels and valid subset for
all per-step duels in that window. The pool sampler visits without replacement
until exhausted, then reshuffles using an independent RNG.

There is **no student control rollout**, no requirement for a generated boxed
answer and no generated-reasoning prefix. The control prompt is question-only
plus the instruction:

```text
Return only the final answer, without reasoning, inside \boxed{}.
```

A non-thinking chat template is followed by the literal runtime cue:

```python
"\nFinal answer: \\boxed{"
```

Here `\n` becomes a real newline and `\\` becomes one LaTeX backslash. Gold boxed
wrappers/headings are canonicalized into answer-content labels. The gold content
is not inserted into the prefix before its prediction position. Only the answer
content tokens are NLL labels; the cue and prompt are not scored as labels.

A control is valid when canonicalization/tokenization succeeds, both prefix and
label tokens are nonempty, and their combined length fits `max_length`.
Malformed boxed gold answers are rejected. At least `min_valid_controls` must
remain; the default eight out of eight is a threshold, not an accuracy test.

The control loss is token-mean NLL per answer, then mean over valid controls:

```math
L_C(\theta)=\frac1{|C_{\rm valid}|}\sum_{c\in C_{\rm valid}}
\frac1{|a_c|}\sum_r-\log p_\theta(a_{c,r}\mid x_c,\tau_{ans},a_{c,<r}).
```

Answer temperature is 1.0 by default. This measures direct-answer prediction,
not correctness of a free-running reasoning completion.

## 7. Exact Virtual AdamW Duel

Hard mode attempts this comparison before **every adaptive optimizer update**.
Soft mode attempts it only at the last step of each window. Both pure gradients
use the entire current main effective batch: 32 examples by default. Eight
`DUEL_PROBLEMS` means eight heldout questions for branch scoring, not eight main
examples for the gradients.

At the same current student/teacher snapshot:

1. Compute detached-target pure gradients `g_F` and `g_M` and aggregate ranks.
2. Save trainable LoRA weights, optimizer state, model buffers/modes and RNG.
3. Evaluate `L_0=L_C(theta)`.
4. Clone LoRA parameters and the actual AdamW state for a FULL-only virtual step.
5. Apply the same current LR, betas, epsilon, weight decay and gradient clipping
   as actual training, then evaluate `L_F=L_C(theta_F_virtual)`.
6. Independently repeat from the original weights/state for MASK, obtaining `L_M`.
7. Restore live LoRA, buffers, modes and RNG, including on exceptions.

The frozen backbone is shared, not copied. Virtual optimizers do not mutate the
live optimizer, scheduler or `.grad`. No hypothetical scheduler step, virtual LR
multiplier, approximate SGD step or optimizer-preconditioned gradient dot product
is substituted for these actual AdamW branches.

The feedback is:

```math
\delta_t=\frac{L_F-L_M}{L_0+\epsilon}.
```

Positive delta favors MASK; negative favors FULL. This is a local, one-step,
heldout surrogate. It is not a causal measurement of end-to-end benchmark gains.
The absolute branch losses and gains are logged. Tiny/zero contrast relative to
FP32 resolution is a diagnostic, not an automatic invalidation or amplification.

## 8. Historical RMS And Hard Leader

Let `v` be an EMA of squared **raw** delta, initially zero. Before ingesting the
current observation, use historical scale and symmetric clipping:

```math
s_t=\max(\sqrt{v_{t-1}},f),\qquad
z_t=\operatorname{clip}(\delta_t/s_t,-c,c),\qquad
D_t=\rho D_{t-1}+z_t.
```

The defaults are `rho=0.9`, floor `f=1e-4`, clip `c=1`. The requested rho=0.8
experiment is a configurable alternative, not a changed global default.

Hard next-step weights are:

```math
w_{M,t+1}=\mathbf1[D_t>0],\qquad w_{F,t+1}=1-w_{M,t+1}.
```

FULL wins an exact tie. There is no sigmoid, exploration floor or permanent
teacher lock in hard mode. `ftrl_eta` and `ftrl_min_weight` are **unused** there.
This is a discounted hard leader on normalized contrasts, not classical
undiscounted FTL on raw cumulative losses.

After using the historical scale, ingest the raw observation. The first valid
observation sets `v=delta^2`; later observations use:

```math
v_t=\beta_{rms}v_{t-1}+(1-\beta_{rms})\delta_t^2.
```

This RMS is **not centered**. A separate centered historical standardizer of raw
delta supplies the actor's lagged state, not the leader's RMS scale.

The actual optimizer step always uses the saved current leader. Duel feedback
is committed after that real update, so it affects only the next step. The
final window step must not update D a second time during actor finalization.

In soft mode, window-end feedback uses the same D recurrence but sets:

```math
w_{M,next}=\varepsilon_w+(1-2\varepsilon_w)\sigma(\eta D),
\qquad w_{F,next}=1-w_{M,next}.
```

Soft defaults are `eta=3`, `epsilon_w=0.02`; these new weights affect the next
window. `fixed_mixture` leaves soft weights fixed and is incompatible with hard.

## 9. Independent Exposure-Actor Reward

The exposure policy is not trained on delta as a reward bonus. Its reward has
two different signals, evaluated over the realized adaptive window of length h.

For each step, record the pre-update main loss `ell_r`. At the last step, cache
that batch's selected targets and rollouts. After the real update, replay the
same frozen target/rollout loss, giving one extra value `ell_{h+1}`. Thus there
are h+1 observations; preceding warmup/window losses are excluded.

At each adaptive step, independently choose four main examples by a separate
checkpointed RNG. Under their selected MASK context, teacher-force the verified
reference labels and record mean log-probability `g_r`. This is **not** a grader
or a generated teacher completion. Log-probabilities are usually negative.

Reference label length defaults to a 1024-token cap. If truncation is necessary,
keep the original reasoning prefix followed by the complete separated answer.
Capacity is bounded independently of sampled mask and answer visibility. If
the complete answer cannot fit, the grounded observation is invalid. The score
is therefore a truncated-reference score when capped; copying bias is possible
because MASK may see retained reference text and the answer.

```math
G_{lp}=\sum_{r=1}^{h}\gamma^{r-1}\max(0,\ell_r-\ell_{r+1}),\qquad
G_{gt}=\frac{\sum_{r=1}^{h}\gamma^{r-1}g_r}{\sum_{r=1}^{h}\gamma^{r-1}},
\qquad R=G_{lp}+\lambda_{gt}G_{gt}.
```

Defaults are `gamma=0.95` and `lambda_gt=0.01`. R is not constrained positive.
Adjacent losses before the last replay are from different batches/targets, and
hard mode can switch teachers. Positive-only loss decreases are a noisy progress
proxy, not a matched before/after causal improvement estimate.

Standardize R using **previous valid history**. For hard mode define the realized
MASK-use fraction `f_M=(1/h)*sum_r w_M,r`; for soft mode use the window's old
fixed MASK weight. The score-function advantage and loss are:

```math
A=f_M Z_R(R),\qquad
L_{actor}=-\operatorname{stopgrad}(A)
\log\pi_\phi(\operatorname{stopgrad}(\alpha)\mid\operatorname{stopgrad}(s)).
```

Realized MASK frequency attenuates actor learning when MASK was rarely used; it
does not solve causal credit assignment. Three **valid actor windows** initialize
statistics before the first policy update. With five FULL warmup steps and
H=5, the earliest actor update is after step 25 if all actor windows are valid.
Leader updates are not delayed by this actor warmup.

Actor Adam LR is 0.01 with gradient norm clip 1.0. There is no entropy bonus,
PPO clipping or KL penalty. Log old-to-new Beta KL on the **same saved state**.
Reward/delta standardizers use centered EMA decay 0.9, variance floors 1e-4/1e-8
and clipping at +/-5. Current samples are ingested only after standardization.
Frozen-controller ablation still samples Beta(2,2) and learns the leader.

## 10. Invalid Feedback Has Two Independent Channels

The student still trains when adaptive feedback is invalid.

| Channel | Representative invalid conditions | Held unchanged |
| --- | --- | --- |
| Expert duel | Too few valid controls; all current masks alias FULL; nonfinite gradients, branch state/losses or delta | D, next leader/weights, RMS, delta standardizer and last-valid delta |
| Actor | Missing/nonfinite loss or grounded observations; all window masks alias FULL; nonfinite R | Policy, reward standardizer, actor valid-window counter and last-valid reward |

An invalid duel does **not** invalidate an otherwise valid actor reward, and an
invalid actor observation does not discard a valid duel. Skipped duels do not
decay D or RMS. Tiny contrast alone is not a skip reason. These redesign rules
differ from the older method that required generated control-prefix probes and
jointly skipped controller/FTRL updates.

## 11. Hard-Mode Execution Order

```python
D = 0
leader = FULL
for actual_optimizer_step in training:
    if FULL_warmup_or_FULL_baseline:
        rollout_from_question_only()
        score_FULL_and_update_student()
        advance_the_same_scheduler()
        continue

    if new_window:
        freeze_direct_answer_controls()
        save_lagged_state_and_sample_detached_alpha()

    old_leader = leader
    y = rollout_from_question_only()
    mask = sample_graph_mask_using_window_alpha()
    q_F, q_M = score_same_y_at_same_pre_step_snapshot()
    g_F, g_M = pure_full_main_batch_gradients(q_F, q_M)
    record_current_loss_and_MASK_reference_logprob()
    duel = isolated_exact_AdamW_duel(g_F, g_M, frozen_controls)

    actual_AdamW_step(g_F if old_leader == FULL else g_M)
    advance_scheduler()
    if last_step_of_window:
        record_post_update_replay_loss_with_frozen_last_batch_targets()

    if duel.valid:
        D, leader = update_historical_RMS_hard_leader_for_next_step(duel)
    if last_step_of_window:
        finalize_actor_reward_and_update_after_valid_warmup()
        # No second leader update here.
    save_checkpoint_if_scheduled_or_final()
```

In the implementation masks are sampled per example, not once for the entire
batch; the pseudocode abbreviates this. The final window may contain fewer than
H actual steps. With a 57-step horizon, warmup 5 and H=5, the last adaptive window
has two steps. Checkpoints at 25, 50 and final step 57 are valid.

## 12. FULL OPSD Comparison And Ablations

Use `variant=full` in this package for the closest controlled FULL-only baseline:
question-only rollouts, same eligible data/main split, FULL privileged context,
same forward-KL reduction, trainable parameters, optimizer and generation recipe.
It bypasses MASK scoring, graph action sampling, controls, duels, actor rewards
and adaptive updates for the whole run. Use the same horizon and checkpoint
steps, rather than continuing a shorter run with a changed LR horizon.

| Comparison | Question it addresses |
| --- | --- |
| Graph hard leader vs matched FULL-only | Does adaptive graph supervision help beyond this FULL distillation recipe? |
| Learned Beta vs frozen Beta(2,2) | Does learned exposure help beyond stochastic graph masking? |
| Graph vs Prefix with matched answer visibility | Does dependency-aware intervention help beyond source-order prefixes? |
| Hard vs soft with matched data/recipe | How does leader selection compare with probability mixing? Cadence/compute also differ. |
| Answer-visible vs answer-hidden MASK | What does the explicit answer block contribute? |
| `lambda_gt=0` vs default | What does the grounded reference term contribute? |

Do not claim equal GPU cost from equal student step counts. Hard mode adds two
pure gradients and three control evaluations per adaptive step, plus grounded
scoring and window-end replay. Report wall time, token counts, forward calls,
replay I/O and multiple-seed variation alongside benchmark results. An actor
mean shift or frequent MASK selection does not by itself establish an accuracy
gain. Do not tune on heldout benchmark test answers.

## 13. Defaults And Adjustable Parameters

These are defaults of `scripts/train_adaptive_graph_ftrl.sh` and
`configs/hard_experiment.yaml`, not every older entrypoint's defaults.

| Environment variable | Config field | Default |
| --- | --- | --- |
| `FTRL_RHO` | `ftrl_rho` | 0.9 |
| `DELTA_RMS_DECAY` | `delta_rms_decay` | 0.9 |
| `DELTA_RMS_FLOOR` | `delta_rms_floor` | 1e-4 |
| `DELTA_SCALED_CLIP` | `delta_scaled_clip` | 1.0 |
| `DUEL_PROBLEMS` | `control_batch_size` | 8 |
| `HELDOUT_PROBLEMS` | `control_pool_size` | 128 |
| `MIN_VALID_CONTROLS` | `min_valid_controls` | 8 |
| `WARMUP_STEPS` | `full_opsd_warmup_steps` | 5 |
| `WINDOW_STEPS` | `window_steps` | 5 |
| `ACTOR_WARMUP_WINDOWS` | `warmup_windows` | 3 |
| `CONTROLLER_LR` | `controller_lr` | 0.01 |
| `EXPOSURE_TEMPERATURE` | `exposure_temperature` | 0.05 |
| `LAMBDA_GT` | `lambda_gt` | 0.01 |
| `GAMMA_DISCOUNT` | `gamma_discount` | 0.95 |
| `LEARNING_RATE` | `learning_rate` | 5e-6 |
| `MAX_STEPS` | `total_updates` | 75 |
| `SAVE_EVERY` | `save_every` | 25 |
| `TRAIN_SEED` / `EVAL_SEED` | `seed` / `eval_seed` | 42 / 42 |

Window size changes alpha/control reuse and actor cadence, **not** hard duel
frequency. The base Python Config uses soft selection, 100 updates and 16
controls; the hard launcher explicitly overrides these. Read the saved run's
`config.yaml` rather than inferring its settings from unrelated defaults.

Shared recipe: Qwen3-4B, BF16 backbone, FP32 LoRA rank 64/alpha 128, no dropout;
four ranks, microbatch 1, accumulation 8, effective batch 32; train generation
temperature 1.1, top-p .95, max new tokens 1024; distillation temperature 1.1;
AdamW betas .9/.999, epsilon 1e-8, weight decay zero, student gradient clip .1.
LR decays linearly over the configured total horizon and is logged **before**
each actual update. FULL warmup is not an LR ramp-up.

Overrides are CLI > explicitly set optional environment override > YAML >
dataclass default. Script-managed schedule/count/seed settings should be changed
via the corresponding environment variables so launch checks match execution.
For example `--ftrl-rho=0.8` is also accepted directly. Invalid settings are
rejected before launching GPU workers. No code edit is needed for these knobs.

## 14. Run, Storage And Evaluation

The server must already have the external training JSONL, model access and the
repository's training/evaluation dependencies. Dataset artifacts, checkpoints,
logs, PDFs and local environments are not part of this algorithm documentation
commit. The redesign now owns its data, runtime and checkpoint helpers; it does
not import the removed legacy methods. Evaluation also needs `eval/`, the two
retained `graph_based_scripts/run_graph_based_eval*.sh` launchers, `pyproject.toml`
and `uv.lock`. The training JSONL remains an external input.

After fetching the new branch on the server:

```bash
cd ~/dropout-self-distillation
git fetch origin
git switch --track origin/adaptive_graph_vs_opsd
source dsd/bin/activate
```

For subsequent updates use `git pull --ff-only origin adaptive_graph_vs_opsd`.
If the local branch already exists, switch to it without `--track`.

Graph hard leader, rho .8, five-step FULL warmup, horizon 57:

```bash
unset RUN_DIR CONFIG
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_ROOT=/research/cbim/vast/qt60/dsd/outputs \
FTRL_RHO=0.8 WARMUP_STEPS=5 \
DUEL_PROBLEMS=8 HELDOUT_PROBLEMS=128 MIN_VALID_CONTROLS=8 \
TRAIN_SEED=42 EVAL_SEED=42 \
MAX_STEPS=57 SAVE_EVERY=25 EVAL_STEPS=25,50,57 \
EVAL_DATASETS=aime25,aime24 EVAL_GPU_MEMORY_UTILIZATION=0.9 \
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh
```

Matched FULL-only baseline, same schedule/data/seed settings:

```bash
unset RUN_DIR CONFIG
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_ROOT=/research/cbim/vast/qt60/dsd/outputs \
FTRL_RHO=0.8 WARMUP_STEPS=5 \
DUEL_PROBLEMS=8 HELDOUT_PROBLEMS=128 MIN_VALID_CONTROLS=8 \
TRAIN_SEED=42 EVAL_SEED=42 \
MAX_STEPS=57 SAVE_EVERY=25 EVAL_STEPS=25,50,57 \
EVAL_DATASETS=aime25,aime24 EVAL_GPU_MEMORY_UTILIZATION=0.9 \
bash adaptive_graph_ftrl_redesign/scripts/train_adaptive_graph_ftrl.sh --variant=full
```

Run these sequentially on the same four GPUs, or use a separate four-GPU group
for a concurrent run. The launcher generates a distinct output directory with
method/parameter/seed labels, config hash, UTC timestamp and PID. The complete
resolved config is frozen to `config.yaml`; `experiment.json` records evaluation
settings. Long filenames are capped, with the remaining settings represented by
the hash. The exact path is printed as `Run:`. Combined output is in
`train_eval.log`; temporary/compiler caches use a short sibling `.runtime`
directory, not the long run name. Nothing is automatically cleaned.

Automatic evaluation checks fresh completion records and final adapter files.
A native teardown error after verified completion can still permit evaluation,
but its nonzero training status is retained. Incomplete training does not trigger
evaluation. Existing checkpoints can also be evaluated with `scripts/run_eval.sh`
and explicit `RUN_DIR`, `EVAL_DATASET` and `EVAL_STEPS`.

Evaluation uses the graph-based/SE launcher recipe: question-only input, thinking
enabled, temperature 1.0, top-p .95, top-k -1, 12 samples per question, up to 38912
new tokens and seed 42. Report **Avg@12, not Pass@12**. Default automatic result
files are under `full_masked_graph_opsd/outputs/evaluation/<run-name>/`; they are
not necessarily in the training run folder. Standalone `run_eval.sh` provides a
configurable `RESULTS_DIR`. Keep all output/temp/cache paths on writable storage;
neither rerouting caches nor this method can guarantee execution when every
available filesystem is full.

## 15. Checkpoints, Logs And Reproducibility

Save atomic adapter/tokenizer checkpoints every `save_every` steps and at the
final endpoint. `training_state.pt` includes student optimizer/scheduler,
controller/Adam, D/weights/RMS/standardizers, valid counters, sampler positions,
RNG streams and pending window state/action/controls/partial observations.
Hard per-step teacher choices survive mid-window resume. Resume is supported
between committed optimizer steps, not within an accumulation microstep.

Changed algorithm definitions, answer visibility, teacher mode, reward settings,
or total training horizon are rejected on ordinary resume. Missing historical
`delta_scaled_clip` is interpreted as the old fixed value 1.0 only. Extending a
completed 50-step run to a different 100-step LR/controller horizon is not
implemented as ordinary exact resume; start a new matched-horizon experiment.

Rank zero shows one progress bar and a compact metric dictionary each step.
Detailed `metrics.jsonl`, `windows.jsonl`, `controls.jsonl` and per-rank sample
logs distinguish current vs next teacher, D, raw/scaled delta, control validity,
RMS scale, loss, LR, mask percentages, reward terms and skip reasons. They also
record teacher calls/token counts, replay I/O, aliases and timing. Window-end
frozen targets are stored temporarily on disk rather than retained across the
window in full-vocabulary GPU memory. This has real disk and I/O cost.

Training/evaluation base seeds default to 42, but independent seeded RNG streams
and per-rank offsets are used. Equal base seeds do not imply bitwise agreement
between different methods, GPU stacks or sampling workloads. Use the saved config
and actual evaluation outputs when reporting results.

## 16. Source Map And Verification Scope

| File | Responsibility |
| --- | --- |
| [train.py](train.py), [cli.py](cli.py) | CLI, data/benchmark exclusion, runtime initialization |
| [trainer.py](trainer.py) | Phases, targets, gradients, controls, rewards and execution order |
| [controller.py](controller.py) | Beta actor, historical statistics and hard/soft leader |
| [duel.py](duel.py) | Isolated exact AdamW branches and state restoration |
| [replay.py](replay.py) | Frozen last-batch target streaming and post-update replay |
| [checkpoint.py](checkpoint.py), [lifecycle.py](lifecycle.py) | Resume, completion proofs and teardown |
| [config.py](config.py), [configs/hard_experiment.yaml](configs/hard_experiment.yaml) | Configuration and defaults |
| [experiment_config.py](experiment_config.py) | Validated snapshots and output naming |
| [scripts/train_adaptive_graph_ftrl.sh](scripts/train_adaptive_graph_ftrl.sh) | Parameterized train-then-eval entrypoint |
| [data.py](data.py), [schema.py](schema.py), [masks.py](masks.py) | Answer split, graph masks and rendering |
| [runtime.py](runtime.py), [model.py](model.py), [rollout.py](rollout.py) | Student runtime, LoRA loading and vLLM snapshot generation |
| [rng.py](rng.py), [sampler.py](sampler.py), [metrics.py](metrics.py) | Checkpointable RNG, data cursors and JSONL logging |
| [evaluation.py](evaluation.py), [eval_adapter.py](eval_adapter.py) | Checkpoint evaluation and existing graph-based launcher adapter |
| [tests/](tests/), [VALIDATION.md](VALIDATION.md) | Tiny-model/unit checks and verification history |

The implementation has CPU tests for detached targets, raw/selected gradients,
exact AdamW branches, independent invalid channels, leader cadence, actor warmup,
checkpoint/resume and parameterized launch/evaluation orchestration. Prior
validation reports 188 combined tests; the rename-specific launcher suite has
26 passing tests. Those checks are not real Qwen3-4B GPU training or an AIME
accuracy result. Actual vLLM memory/teardown, server dataset contents, throughput
and benchmark performance require a server experiment.
