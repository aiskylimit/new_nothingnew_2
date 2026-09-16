"""TROPIC-G (TROPIC-P + Eq. 11's classifier-free-guidance debiasing, alpha>0),
TOP-K=64 (sparsified) variant, as an offline-deployment baseline, run under
the SAME experimental setting this offline package already uses for
RLSD/SDPO (see scripts/run_rlsd_experiment_4b.py / run_sdpo_experiment_4b.py):
same base model (Qwen/Qwen3-4B, overridable via --model-path for a local
offline directory), same LoRA config (r=64, alpha=128, same target modules),
same lr=5e-6/max_grad_norm=0.1/gradient_checkpointing, same checkpoint cadence
(20/25/40/50/60/75/80/100), same eval protocol (30 problems x k=12 samples x
{aime25,aime26,hmmt25} by default), same unrestricted train+eval sampling
recipe (temperature=1.0/top_p=1.0/top_k=-1), same --skip-train/--skip-eval/
--model-path/--vllm-gpu-memory-utilization/--training-steps offline
mechanisms, same TROPIC_TRAIN_DATA_PATH/TROPIC_EVAL_DATA_DIR env-var
overrides (handled inside tropic/data.py and tropic/eval.py - this script
never references those env vars directly).

SIBLING of run_tropic_g_experiment_4b.py (the FULL-VOCABULARY variant already
in this folder) - byte-identical in every setting EXCEPT `top_k` in
CONFIG["tropic_g"] (64 here vs 200000 there) and the tag/output-dir (so the
two never clobber each other's results: TAG="tropic_g_topk64",
--output-dir default "results_tropic_g_topk64_4b"). Added per explicit
request to also run the ORIGINAL sparsified top_k=64 tropic_g (matching the
ONLINE cluster's own tropic_g default) on this offline B200 deployment,
alongside the full-vocab ablation already there - both loaded from the SAME
already-downloaded model, same tropic/*.py modules, no new dependencies.

DELTAS from the shared RLSD/SDPO setting, exactly as agreed with the user
before this file was written:
  - alpha=0.25 (Eq. 11's debias weight - same value the ONLINE cluster's
    tropic_g/tropic_k/tropic_adaptive scripts use).
  - beta=0.1, epsilon=0.1 (Eq. 13-16's fixed trust-region radius and Eq. 6's
    skew - same values every ONLINE tropic_p/tropic_g/tropic_k script uses;
    NOT the adaptive schedule tropic_adaptive/tropic_g_adaptive use).
  - top_k=64 in CONFIG["tropic_g"] - the ONLINE cluster's own default
    sparsification width for tropic_g/tropic_k/tropic_adaptive (NOT full
    vocabulary - see run_tropic_g_experiment_4b.py in this same folder for
    that ablation instead).
  - effective_batch_size=16 (this offline package's own choice, matching
    RLSD/SDPO's FINAL reduced rollout budget on this untested-on-B200
    pipeline for the 4B model: group_size=4 x questions_per_step=4=16 there;
    TROPIC-G has no group_size concept at all - each micro-example is one
    (question, single-rollout) pair, not a GRPO-style group - so 16 here
    means 16 independent (question, rollout) pairs per training step,
    NOT 16 rollouts drawn from 4 grouped questions like RLSD/SDPO's 4x4).
    The 8B script (run_tropic_g_experiment_8b.py) uses effective_batch_size=4,
    matching RLSD/SDPO's 8B budget (group_size=2 x questions_per_step=2=4)
    the same way.

Everything else (only_correct=True, teacher_thinking=True,
student_thinking=False, fixed_teacher=True - OPSD's own real recipe, teacher
frozen at initial weights - max_length=20000, training_steps=100 default)
matches the ONLINE cluster's own tropic_g scripts exactly, per "mọi setting
thực nghiệm giống 2 bài kia [RLSD/SDPO], trừ [the deltas above]".

This is a STANDALONE script - no existing tropic/*.py module in this offline
package is modified for it. It reuses the EXISTING, unmodified
`tropic.model.ContextualPolicy`, `tropic.rollout.generate_rollout`,
`tropic.loss.tropic_p_loss`/`privileged_gap`, `tropic.primitives.
debias_logits`/`normalize_log`, `tropic.data.load_opsd_math_examples`/
`build_teacher_context_only_prefix`, and `tropic.eval` (benchmark loading,
math_verify-based grading, scoring) - the exact same building blocks the
ONLINE cluster's run_tropic_adaptive_4b.py's "tropic_g" branch uses, copied
here as a standalone offline script the same way run_rlsd_experiment_4b.py
and run_sdpo_experiment_4b.py were.

Usage - one-time training (no eval), THEN eval per checkpoint step (matches
run_rlsd_experiment_4b.py's own usage pattern exactly):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_tropic_g_topk64_4b.py \\
        --model-path /path/to/local/Qwen3-4B --seed 0 --output-dir results_tropic_g_topk64_4b --skip-eval \\
        --use-vllm-rollout --vllm-gpu-ids 3 --vllm-base-port 8100
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_tropic_g_topk64_4b.py \\
        --model-path /path/to/local/Qwen3-4B --skip-train \\
        --checkpoint-path results_tropic_g_topk64_4b/tropic_g_topk64_checkpoint_step100 \\
        --output-dir results_tropic_g_topk64_4b --eval-engine vllm --vllm-gpu-ids 1,2,3 --vllm-base-port 8100
"""
import argparse
import json
import re
import time
import gc
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="results_tropic_g_topk64_4b",
                     help="Where to write results/checkpoints.")
parser.add_argument("--seed", type=int, default=0,
                     help="Seed for the training-example draw AND torch's global RNG.")
parser.add_argument("--eval-benchmarks", nargs="+", default=["aime25", "aime26", "hmmt25"],
                     choices=["aime24", "aime25", "aime26", "hmmt25"])
parser.add_argument("--model-path", default=None,
                     help="Local directory holding a raw download of the base model (offline "
                          "deployment - e.g. a no-internet server pre-populated via `download.txt`) "
                          "- overrides CONFIG['model_name']. Omit to use the normal online repo id.")
parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                     help="Passed through to `vllm serve --gpu-memory-utilization` for every "
                          "replica (train and eval). Omit to use vLLM's own default (0.9).")
parser.add_argument("--training-steps", type=int, default=100,
                     help="Overrides CONFIG['train']['training_steps'] (default 100) - lower this "
                          "for a quick dry run (e.g. 3) before committing to a full run.")
parser.add_argument("--alpha", type=float, default=0.25,
                     help="Eq. 11's classifier-free-guidance debias weight - default 0.25, same as "
                          "the online cluster's tropic_g/tropic_k/tropic_adaptive scripts.")
parser.add_argument("--eval-engine", choices=["vllm", "hf"], default="hf",
                     help="Backend for eval generation. 'hf' (default) uses plain "
                          "transformers.generate(). 'vllm' generates via data-parallel vLLM "
                          "server replicas (tropic.vllm_rollout_client) - requires "
                          "--vllm-gpu-ids (free GPUs, disjoint from this process's own "
                          "CUDA_VISIBLE_DEVICES); several times faster for this workload (many "
                          "long sequences).")
parser.add_argument("--vllm-gpu-ids", default=None,
                     help="Comma-separated GPU ids to run vLLM eval replicas on (e.g. '5,6,7') - "
                          "MUST be free GPUs, disjoint from this process's own CUDA_VISIBLE_DEVICES "
                          "and from any other job. Required with --eval-engine vllm.")
parser.add_argument("--vllm-base-port", type=int, default=8100,
                     help="First port for the vLLM replica servers (replica i uses base_port+i) - "
                          "use a DIFFERENT range per concurrently-running vLLM-enabled job on the "
                          "same machine to avoid port collisions.")
parser.add_argument("--vllm-executable", default="vllm",
                     help="Path to the `vllm` CLI - pass an ABSOLUTE PATH when vLLM lives in a "
                          "different conda env than this process (see "
                          "tropic/vllm_rollout_client.py's docstring for why that's usually needed).")
parser.add_argument("--use-vllm-rollout", action="store_true",
                     help="Generate training rollouts via data-parallel vLLM server replicas "
                          "(tropic.vllm_rollout_client) instead of sequential HF model.generate() "
                          "calls. Requires --vllm-gpu-ids.")
parser.add_argument("--skip-train", action="store_true",
                     help="Skip training; load an already-trained checkpoint for eval instead "
                          "(requires --checkpoint-path).")
parser.add_argument("--checkpoint-path", default=None)
parser.add_argument("--skip-eval", action="store_true",
                     help="Train and checkpoint as usual, but skip eval - eval is instead run "
                          "separately by other processes pointed at the saved checkpoints.")
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
args = parser.parse_args()
if args.skip_train and not args.checkpoint_path:
    parser.error("--skip-train requires --checkpoint-path")
if args.use_vllm_rollout and not args.vllm_gpu_ids:
    parser.error("--use-vllm-rollout requires --vllm-gpu-ids")

t_start = time.time()
RESULTS_DIR = Path(__file__).resolve().parent.parent / args.output_dir
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT_STEPS = {20, 25, 40, 50, 60, 75, 80, 100}  # same union cadence as RLSD/SDPO's offline scripts


def tick(label):
    print(f"[{time.time() - t_start:8.1f}s] {label}", flush=True)


tick("import torch/transformers/datasets/peft")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

torch.manual_seed(args.seed)
tick(f"imports done, torch.manual_seed({args.seed})")

LORA_CONFIG = dict(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

CONFIG = dict(
    model_name=args.model_path or "Qwen/Qwen3-4B",
    only_correct=True, teacher_thinking=True, student_thinking=False,
    fixed_teacher=True,  # OPSD's own real recipe: teacher = frozen initial policy (LoRA disabled)
    max_length=20000,
    lora=LORA_CONFIG,
    train=dict(training_steps=args.training_steps, effective_batch_size=16, lr=5e-6, max_grad_norm=0.1,
               max_new_tokens=1024, temperature=1.0, top_p=1.0, top_k=-1, seed=args.seed,
               gradient_checkpointing=True),
    # epsilon=0.1/beta=0.1/alpha=0.25/top_k=64: same fixed values as the
    # online cluster's own tropic_g default (sparsified top-K, NOT the
    # full-vocab ablation in run_tropic_g_experiment_4b.py).
    tropic_g=dict(epsilon=0.1, beta=0.1, top_k=64, default_mass=1e-5, alpha=args.alpha),
    eval=dict(benchmarks=args.eval_benchmarks, num_problems=30, k=12,
              max_new_tokens=38912, temperature=1.0, top_p=1.0, top_k=-1,
              min_p=0.0, enable_thinking=True, gen_batch_size=4),
)
print(json.dumps(CONFIG, indent=2), flush=True)

tick("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tick("tokenizer loaded")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
tick(f"device={device} dtype={dtype}")
if not torch.cuda.is_available():
    print("WARNING: no GPU detected - this config will be extremely slow on CPU.", flush=True)


def load_fresh_model():
    try:
        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"], dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device)
    except (ImportError, ValueError) as e:
        tick(f"flash_attention_2 unavailable ({e}); falling back to default attention implementation")
        base = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=dtype).to(device)
    if CONFIG["train"].get("gradient_checkpointing", False):
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        base.config.use_cache = False
    lora_cfg = LoraConfig(task_type="CAUSAL_LM", **CONFIG["lora"])
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_model_from_checkpoint(checkpoint_path):
    try:
        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"], dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device)
    except (ImportError, ValueError) as e:
        tick(f"flash_attention_2 unavailable ({e}); falling back to default attention implementation")
        base = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    tick(f"loaded checkpoint from {checkpoint_path}")
    return model


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ALL_BENCHMARKS = {"aime24", "aime25", "aime26", "hmmt25"}


def infer_step_suffix(checkpoint_path):
    if not checkpoint_path:
        return ""
    m = re.search(r"_checkpoint_step(\d+)/?$", str(checkpoint_path).rstrip("/\\"))
    return f"_step{m.group(1)}" if m else ""


def bench_suffix_for(benchmarks):
    if set(benchmarks) == ALL_BENCHMARKS:
        return ""
    return "_" + "-".join(sorted(benchmarks))


from tropic.data import build_teacher_context_only_prefix, load_opsd_math_examples
from tropic.model import ContextualPolicy
from tropic.rollout import generate_rollout
from tropic.loss import privileged_gap, tropic_p_loss
from tropic.primitives import debias_logits, normalize_log
from tropic.eval import load_benchmark, score_generations
from tropic.vllm_rollout_client import (
    generate_eval_batch_parallel, generate_rollout_batch_parallel,
    launch_vllm_replicas, refresh_lora_adapter, shutdown_vllm_replicas,
)

train_examples = None
if not args.skip_train:
    num_train_examples = CONFIG["train"]["training_steps"] * CONFIG["train"]["effective_batch_size"]
    tick(f"loading + filtering OPSD dataset, num_examples={num_train_examples}")
    train_examples = load_opsd_math_examples(
        tokenizer,
        num_examples=num_train_examples,
        only_correct=CONFIG["only_correct"],
        teacher_thinking=CONFIG["teacher_thinking"],
        student_thinking=CONFIG["student_thinking"],
        seed=CONFIG["train"]["seed"],
        max_length=CONFIG["max_length"],
    )
    tick(f"dataset ready, {len(train_examples)} examples")


def train_run_tropic_g(model, examples, cfg, tag, use_vllm_rollout=False, vllm_gpu_ids=None,
                        vllm_base_port=8100, vllm_executable="vllm"):
    policy = ContextualPolicy(model, tokenizer, fixed_teacher=cfg.get("fixed_teacher", False))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    tick(f"[{tag}] optimizer over {n_trainable} trainable params")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["train"]["lr"])
    training_steps = cfg["train"]["training_steps"]
    effective_batch_size = cfg["train"]["effective_batch_size"]
    example_idx = 0
    logs = []
    run_start = time.time()

    vllm_procs, vllm_ports, vllm_scratch_dir = None, None, None
    if use_vllm_rollout:
        vllm_scratch_dir = RESULTS_DIR / f"{tag}_vllm_scratch_adapter"
        tick(f"[{tag}] launching {len(vllm_gpu_ids)} vLLM rollout replica(s) on GPU {vllm_gpu_ids}")
        vllm_procs, vllm_ports = launch_vllm_replicas(
            cfg["model_name"], vllm_gpu_ids, vllm_base_port, cfg["lora"]["r"],
            log_dir=str(RESULTS_DIR), vllm_executable=vllm_executable,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
        model.save_pretrained(str(vllm_scratch_dir))
        refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))
        tick(f"[{tag}] vLLM replicas ready, initial adapter loaded")

    try:
        for step in range(training_steps):
            t0 = time.time()
            optimizer.zero_grad()
            micro_losses, micro_s, micro_g = [], [], []
            last_ex = last_generated_ids = last_log_old = None

            step_examples = [examples[(example_idx + i) % len(examples)] for i in range(effective_batch_size)]
            example_idx += effective_batch_size
            if use_vllm_rollout:
                vllm_rollouts = generate_rollout_batch_parallel(
                    vllm_ports, [ex.student_prompt_ids for ex in step_examples], tokenizer,
                    max_new_tokens=cfg["train"]["max_new_tokens"], temperature=cfg["train"]["temperature"],
                    top_p=cfg["train"]["top_p"], top_k=cfg["train"].get("top_k"),
                )

            for micro in range(effective_batch_size):
                ex = step_examples[micro]
                if use_vllm_rollout:
                    generated_ids, gen_len = vllm_rollouts[micro]
                    generated_ids = generated_ids.to(model.device)
                else:
                    generated_ids, gen_len = generate_rollout(
                        model, tokenizer, ex.student_prompt_ids,
                        max_new_tokens=cfg["train"]["max_new_tokens"],
                        temperature=cfg["train"]["temperature"], top_p=cfg["train"]["top_p"],
                        top_k=cfg["train"].get("top_k"),
                    )
                log_old = policy.forward_checkpoint(ex.student_prompt_ids, generated_ids)
                log_student = policy.forward_student(ex.student_prompt_ids, generated_ids)
                log_teacher = policy.forward_teacher_logits(ex.teacher_prefix_ids, generated_ids)
                context_only_prefix_ids = build_teacher_context_only_prefix(
                    tokenizer, ex.reference_solution,
                    enable_thinking=CONFIG["teacher_thinking"], max_length=CONFIG["max_length"],
                )
                log_teacher_context_only = policy.forward_teacher_context_only(context_only_prefix_ids, generated_ids)
                loss, s = tropic_p_loss(
                    log_student, log_teacher, log_old, generated_ids,
                    teacher_context_only_logits=log_teacher_context_only, **cfg["tropic_g"],
                )
                micro_s.append(s.mean().item())
                pi_bar_teacher_log = normalize_log(
                    debias_logits(log_teacher, log_teacher_context_only, cfg["tropic_g"]["alpha"])
                )
                micro_g.append(privileged_gap(log_old, pi_bar_teacher_log).item())

                (loss / effective_batch_size).backward()
                micro_losses.append(loss.item())
                last_ex, last_generated_ids, last_log_old = ex, generated_ids, log_old

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg["train"]["max_grad_norm"])
            optimizer.step()

            if use_vllm_rollout:
                model.save_pretrained(str(vllm_scratch_dir))
                refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))

            mean_loss = sum(micro_losses) / len(micro_losses)
            record = {"tag": tag, "step": step, "loss": mean_loss, "elapsed_sec": time.time() - t0,
                      "mean_s": sum(micro_s) / len(micro_s), "G_k": sum(micro_g) / len(micro_g)}
            logs.append(record)

            avg_sec = (time.time() - run_start) / (step + 1)
            record["eta_min"] = round(avg_sec * (training_steps - step - 1) / 60.0, 1)
            tick(f"[{tag}] {json.dumps(record)}")
            if mean_loss != mean_loss:
                raise RuntimeError(f"[{tag}] loss became NaN at step={step}")

            if (step + 1) in CHECKPOINT_STEPS:
                ckpt_dir = RESULTS_DIR / f"{tag}_checkpoint_step{step + 1}"
                model.save_pretrained(str(ckpt_dir))
                tick(f"[{tag}] checkpoint saved: {ckpt_dir}")

            del last_generated_ids, last_log_old
            if step % 10 == 0:
                clear_cuda_cache()
    finally:
        if use_vllm_rollout:
            shutdown_vllm_replicas(vllm_procs)
            tick(f"[{tag}] vLLM rollout replicas shut down")

    optimizer.zero_grad(set_to_none=True)
    del optimizer, policy
    clear_cuda_cache()
    (RESULTS_DIR / f"{tag}_train_logs.json").write_text(json.dumps(logs, indent=2))
    tick(f"[{tag}] training done, logs saved to {RESULTS_DIR}/{tag}_train_logs.json")
    return logs


@torch.no_grad()
def generate_k_samples(model, problem_text, k, cfg_eval):
    model.eval()
    messages = [{"role": "user", "content": problem_text + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=cfg_eval["enable_thinking"])
    input_ids = ids["input_ids"] if hasattr(ids, "__getitem__") and not torch.is_tensor(ids) else ids
    input_ids = input_ids.to(model.device)
    top_k_cfg = cfg_eval.get("top_k", 50)
    top_k_hf = 0 if top_k_cfg is None or top_k_cfg < 0 else top_k_cfg
    gen_batch_size = cfg_eval.get("gen_batch_size", k)

    completions = []
    remaining = k
    while remaining > 0:
        this_batch = min(gen_batch_size, remaining)
        batch = input_ids.repeat(this_batch, 1)
        attention_mask = torch.ones_like(batch)
        out = model.generate(
            input_ids=batch, attention_mask=attention_mask,
            max_new_tokens=cfg_eval["max_new_tokens"], do_sample=True,
            temperature=cfg_eval["temperature"], top_p=cfg_eval["top_p"],
            top_k=top_k_hf, min_p=cfg_eval.get("min_p", 0.0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completions.extend(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True))
        del batch, attention_mask, out
        clear_cuda_cache()
        remaining -= this_batch
    return completions


def evaluate_model(model, cfg, tag, shard_index=0, num_shards=1):
    suffix = f"_shard{shard_index}of{num_shards}" if num_shards > 1 else ""
    results = {}
    all_generations = {}
    for bench_name in cfg["eval"]["benchmarks"]:
        problems = load_benchmark(bench_name, num_problems=cfg["eval"]["num_problems"])
        if num_shards > 1:
            problems = problems[shard_index::num_shards]
        generations, golds = [], []
        for i, p in enumerate(problems):
            tick(f"[{tag}{suffix}] {bench_name} problem {i+1}/{len(problems)} (id={p.problem_id})")
            gens = generate_k_samples(model, p.problem, cfg["eval"]["k"], cfg["eval"])
            generations.append(gens)
            golds.append(p.answer)
        all_generations[bench_name] = {
            "problem_ids": [p.problem_id for p in problems],
            "golds": golds,
            "generations": generations,
        }
        if num_shards == 1:
            res = score_generations(bench_name, generations, golds)
            results[bench_name] = res
            tick(f"[{tag}] {bench_name} avg@{res.k}={res.avg_at_k:.3f} pass@{res.k}={res.pass_at_k:.3f}")
        else:
            tick(f"[{tag}{suffix}] {bench_name} done ({len(problems)} problems in this shard)")

    (RESULTS_DIR / f"{tag}{suffix}_generations.json").write_text(json.dumps(all_generations, indent=2))
    if num_shards == 1:
        serializable = {k: {"avg_at_k": v.avg_at_k, "pass_at_k": v.pass_at_k, "k": v.k} for k, v in results.items()}
        (RESULTS_DIR / f"{tag}_eval_results.json").write_text(json.dumps(serializable, indent=2))
    tick(f"[{tag}{suffix}] eval done - raw generations saved under {RESULTS_DIR}/")
    return results


def evaluate_model_vllm(checkpoint_dir, cfg, tag, vllm_gpu_ids, vllm_base_port=8100, vllm_executable="vllm"):
    from tropic.vllm_rollout_client import (
        generate_eval_batch_parallel, launch_vllm_replicas, refresh_lora_adapter, shutdown_vllm_replicas,
    )

    tick(f"[{tag}] launching {len(vllm_gpu_ids)} vLLM eval replica(s) on GPU {vllm_gpu_ids} "
         f"(ports {vllm_base_port}..{vllm_base_port + len(vllm_gpu_ids) - 1})")
    procs, ports = launch_vllm_replicas(
        cfg["model_name"], vllm_gpu_ids, vllm_base_port, cfg["lora"]["r"],
        log_dir=str(RESULTS_DIR), vllm_executable=vllm_executable,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    try:
        refresh_lora_adapter(ports, str(checkpoint_dir))
        tick(f"[{tag}] vLLM eval replicas ready, checkpoint adapter loaded from {checkpoint_dir}")

        results = {}
        all_generations = {}
        top_k_cfg = cfg["eval"].get("top_k", -1)
        for bench_name in cfg["eval"]["benchmarks"]:
            problems = load_benchmark(bench_name, num_problems=cfg["eval"]["num_problems"])
            prompts = []
            for p in problems:
                messages = [{"role": "user", "content": p.problem + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=cfg["eval"]["enable_thinking"],
                )
                prompts.append(text)

            tick(f"[{tag}] {bench_name}: generating {len(prompts)} problems x {cfg['eval']['k']} samples via vLLM")
            generations = generate_eval_batch_parallel(
                ports, prompts,
                max_tokens=cfg["eval"]["max_new_tokens"], temperature=cfg["eval"]["temperature"],
                top_p=cfg["eval"]["top_p"], n=cfg["eval"]["k"],
                top_k=top_k_cfg if top_k_cfg and top_k_cfg > 0 else None,
                min_p=cfg["eval"].get("min_p", 0.0) or None,
            )
            golds = [p.answer for p in problems]

            res = score_generations(bench_name, generations, golds)
            results[bench_name] = res
            all_generations[bench_name] = {
                "problem_ids": [p.problem_id for p in problems],
                "golds": golds,
                "generations": generations,
            }
            tick(f"[{tag}] {bench_name} avg@{res.k}={res.avg_at_k:.3f} pass@{res.k}={res.pass_at_k:.3f}")

            (RESULTS_DIR / f"{tag}_generations.json").write_text(json.dumps(all_generations, indent=2))
            serializable = {k: {"avg_at_k": v.avg_at_k, "pass_at_k": v.pass_at_k, "k": v.k} for k, v in results.items()}
            (RESULTS_DIR / f"{tag}_eval_results.json").write_text(json.dumps(serializable, indent=2))

        tick(f"[{tag}] eval done (vLLM) - scores + raw generations saved under {RESULTS_DIR}/")
        return results
    finally:
        shutdown_vllm_replicas(procs)
        tick(f"[{tag}] vLLM eval replicas shut down")


TAG = "tropic_g_topk64"  # distinct from the full-vocab script's "tropic_g" tag, so results/checkpoints never mix
EVAL_TAG = TAG + infer_step_suffix(args.checkpoint_path if args.skip_train else None) + bench_suffix_for(args.eval_benchmarks)

if args.skip_train:
    tick(f"=== {TAG}: --skip-train, loading checkpoint {args.checkpoint_path} ===")
    model = load_model_from_checkpoint(args.checkpoint_path)
else:
    model = load_fresh_model()
    model.train()
    vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
    train_run_tropic_g(model, train_examples, CONFIG, TAG,
                        use_vllm_rollout=args.use_vllm_rollout, vllm_gpu_ids=vllm_gpu_ids,
                        vllm_base_port=args.vllm_base_port, vllm_executable=args.vllm_executable)
    if args.skip_eval:
        tick(f"=== {TAG}: --skip-eval, training done, NOT evaluating in this process. "
             f"Checkpoints saved under {RESULTS_DIR}/. Now launch eval with "
             f"--skip-train --checkpoint-path {RESULTS_DIR}/{TAG}_checkpoint_step<N>.")
        sys.exit(0)

if args.eval_engine == "hf":
    result = evaluate_model(model, CONFIG, EVAL_TAG, shard_index=args.shard_index, num_shards=args.num_shards)
    del model
    clear_cuda_cache()
else:
    checkpoint_dir_for_vllm = (
        Path(args.checkpoint_path) if args.skip_train
        else RESULTS_DIR / f"{TAG}_checkpoint_step{CONFIG['train']['training_steps']}"
    )
    del model
    clear_cuda_cache()
    tick(f"model_{TAG} freed - GPU clear for vLLM")
    eval_vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
    if not eval_vllm_gpu_ids:
        parser.error("--eval-engine vllm requires --vllm-gpu-ids")
    result = evaluate_model_vllm(checkpoint_dir_for_vllm, CONFIG, EVAL_TAG, eval_vllm_gpu_ids,
                                  args.vllm_base_port, args.vllm_executable)

if args.num_shards > 1:
    step_num = infer_step_suffix(args.checkpoint_path).lstrip("_step") or "<N>"
    tick(f"ALL DONE (eval shard {args.shard_index}/{args.num_shards}) - once all shards finish: "
         f"python scripts/combine_shards.py --output-dir {args.output_dir} --tag {TAG} "
         f"--num-shards {args.num_shards} --checkpoint-step {step_num} --benchmarks {' '.join(args.eval_benchmarks)}")
    sys.exit(0)

tick(f"ALL DONE - results saved under {RESULTS_DIR}/")
