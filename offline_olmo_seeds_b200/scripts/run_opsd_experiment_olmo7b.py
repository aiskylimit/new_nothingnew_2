"""OPSD on OLMo-3-7B-Think with this offline_olmo7b_b200 package (TAG="opsd_olmo", --output-dir default
"results_opsd_olmo7b").

Built from run_tropic_g_topk64_olmo7b.py so that everything except the loss uses this folder's Olmo setup:
OLMo prompts (empty_think_suffix for the non-thinking student), data order (seed), rollout sampling
(temperature=1.0 / top_p=1.0 / top_k=-1), effective batch 32, LoRA, lr, grad clip, checkpoint grid, and eval
(temperature=1.0 / top_p=0.95, avg@12) - all as in paper Table 10.

Loss = OPSD's official recipe (as in scripts/run_opsd_8b_tokeff.py): frozen initial teacher reading the
reference solution, forward KL (beta=0) over the full vocabulary, per-token clip 0.06 (--jsd-token-clip),
temperature 1.1 inside the loss (tropic.baselines.opsd_baseline_loss).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_opsd_experiment_olmo7b.py --seed 1 \
        --output-dir results_olmo7b_opsd_seed1 --skip-eval --use-vllm-rollout --vllm-gpu-ids 1,2,3
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_opsd_experiment_olmo7b.py --skip-train \
        --checkpoint-path results_olmo7b_opsd_seed1/opsd_olmo_checkpoint_step25 \
        --output-dir results_olmo7b_opsd_seed1 --eval-engine vllm --vllm-gpu-ids 0,1,2,3
"""
import argparse
import json
import re
import time
import gc
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="results_opsd_olmo7b",
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
parser.add_argument("--vllm-max-model-len", type=int, default=41000,
                     help="Passed through to `vllm serve --max-model-len` for every replica "
                          "(train and eval). Without this, vLLM auto-detects the model's own "
                          "max_position_embeddings (can be far larger than anything this project "
                          "ever sends it), which makes it reserve per-sequence KV cache for a "
                          "context length nothing reaches - artificially capping how many "
                          "sequences can run concurrently even with plenty of free VRAM. Default "
                          "41000 covers eval's max_new_tokens=38912 plus a short prompt; pass 0 "
                          "to fall back to vLLM's own auto-detection.")
parser.add_argument("--training-steps", type=int, default=100,
                     help="Overrides CONFIG['train']['training_steps'] (default 100) - lower this "
                          "for a quick dry run (e.g. 3) before committing to a full run.")
parser.add_argument("--jsd-token-clip", type=float, default=0.06,
                     help="OPSD per-token clip; 0.06 = OPSD's official 8B recipe (run_opsd_8b.sh).")
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
parser.add_argument("--effective-batch-size", type=int, default=32,
                     help="Rollouts per optimizer step. Default 32 = paper Table 10 (same as the Qwen3-8B runs); "
                          "the original offline Olmo setting was 4.")
args = parser.parse_args()
if args.skip_train and not args.checkpoint_path:
    parser.error("--skip-train requires --checkpoint-path")
if args.use_vllm_rollout and not args.vllm_gpu_ids:
    parser.error("--use-vllm-rollout requires --vllm-gpu-ids")

t_start = time.time()
RESULTS_DIR = Path(__file__).resolve().parent.parent / args.output_dir
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT_STEPS = {10, 15, 20, 25, 40, 50, 60, 75, 80, 100}  # same union cadence as RLSD/SDPO's offline scripts, plus 10/15


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
    model_name=args.model_path or "allenai/Olmo-3-7B-Think",
    only_correct=True, teacher_thinking=True, student_thinking=False,
    fixed_teacher=True,  # OPSD's own real recipe: teacher = frozen initial policy (LoRA disabled)
    max_length=20000,
    # OLMo-3-7B-Think has no native enable_thinking toggle - this closes its
    # auto-opened <think> tag empty when enable_thinking=False (student only;
    # eval and the thinking teacher never see this) - see this file's own
    # docstring (UNVERIFIED end-to-end) and tropic/data.py's docstring.
    empty_think_suffix="\n\n</think>\n\n",
    lora=LORA_CONFIG,
    train=dict(training_steps=args.training_steps, effective_batch_size=args.effective_batch_size, lr=5e-6, max_grad_norm=0.1,
               # Rollout sampling as in paper Table 10: 1.0 / 1.0 / -1.
               max_new_tokens=1024, temperature=1.0, top_p=1.0, top_k=-1, seed=args.seed,
               gradient_checkpointing=True),
    # OPSD's official recipe (as in scripts/run_opsd_8b_tokeff.py): forward KL (beta=0), full
    # vocabulary, temperature 1.1 inside the loss, per-token clip 0.06.
    opsd_baseline=dict(beta=0.0, jsd_token_clip=args.jsd_token_clip, temperature=1.1, top_k=None, default_mass=None),
    # EVAL sampling as in paper Table 10: temperature=1.0, top_p=0.95 (same as the Qwen3-8B runs).
    eval=dict(benchmarks=args.eval_benchmarks, num_problems=30, k=12,
              max_new_tokens=38912, temperature=1.0, top_p=0.95, top_k=-1,
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


from tropic.data import load_opsd_math_examples
from tropic.model import ContextualPolicy
from tropic.rollout import generate_rollout
from tropic.baselines import opsd_baseline_loss
from tropic.primitives import kl_categorical
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
        empty_think_suffix=CONFIG["empty_think_suffix"],
    )
    tick(f"dataset ready, {len(train_examples)} examples")


def train_run_opsd(model, examples, cfg, tag, use_vllm_rollout=False, vllm_gpu_ids=None,
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
            max_model_len=(args.vllm_max_model_len or None),
        )
        model.save_pretrained(str(vllm_scratch_dir))
        refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))
        tick(f"[{tag}] vLLM replicas ready, initial adapter loaded")

    try:
        for step in range(training_steps):
            t0 = time.time()
            optimizer.zero_grad()
            micro_losses = []
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
                log_teacher = policy.forward_teacher(ex.teacher_prefix_ids, generated_ids)  # frozen initial model
                loss = opsd_baseline_loss(log_student, log_teacher, generated_ids, **cfg["opsd_baseline"])

                (loss / effective_batch_size).backward()
                micro_losses.append(loss.item())
                last_ex, last_generated_ids, last_log_old = ex, generated_ids, log_old

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg["train"]["max_grad_norm"])
            optimizer.step()

            if use_vllm_rollout:
                model.save_pretrained(str(vllm_scratch_dir))
                refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))

            mean_loss = sum(micro_losses) / len(micro_losses)
            with torch.no_grad():
                drift = kl_categorical(policy.forward_student(last_ex.student_prompt_ids, last_generated_ids),
                                       last_log_old).mean().item()
            record = {"tag": tag, "step": step, "loss": mean_loss, "elapsed_sec": time.time() - t0,
                      "kl_student_old_drift": drift}
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
                # Also persist the per-step logs at every checkpoint, so a crash mid-run keeps them.
                (RESULTS_DIR / f"{tag}_train_logs.json").write_text(json.dumps(logs, indent=2))

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
        max_model_len=(args.vllm_max_model_len or None),
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


TAG = "opsd_olmo"
EVAL_TAG = TAG + infer_step_suffix(args.checkpoint_path if args.skip_train else None) + bench_suffix_for(args.eval_benchmarks)

if args.skip_train:
    tick(f"=== {TAG}: --skip-train, loading checkpoint {args.checkpoint_path} ===")
    model = load_model_from_checkpoint(args.checkpoint_path)
else:
    model = load_fresh_model()
    model.train()
    vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
    train_run_opsd(model, train_examples, CONFIG, TAG,
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
