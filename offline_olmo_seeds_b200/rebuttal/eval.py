"""Evaluate saved checkpoints of one run with vLLM, one JSON per (step, benchmark).

Writes runs/<model>/<method>/seed<k>/eval/step<N>/<bench>.json with, for every
problem, the correctness / token length / finish reason / extracted answer of
each of the k samples (enough for avg@k, pass@k, bootstrap CIs, length and
answer-entropy statistics), and the raw texts in <bench>.texts.json.gz for
re-grading. Finished (step, benchmark) files are skipped, so the job can be
re-launched after a crash.

Every saved checkpoint (10/15/20/25/40/50/75/100) is evaluated on every
benchmark, for every method and seed, so analyze.py can apply the paper's
protocol (best checkpoint per benchmark) per seed, and also report the final
and the best single checkpoint.

Usage (all four GPUs serve vLLM replicas; nothing is loaded in this process):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python rebuttal/eval.py \\
        --model qwen3-8b --method tropic --seed 0 \\
        --vllm-gpu-ids 0,1,2,3 --vllm-base-port 8100 --vllm-executable /path/to/tropic-vllm/bin/vllm

    # untrained base model (step 0), shared by all methods:
    ... rebuttal/eval.py --model qwen3-8b --base ...
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal.presets import (  # noqa: E402
    EVAL_K, EVAL_MAX_NEW_TOKENS, EVAL_NUM_PROBLEMS, LORA, MODEL_PRESETS, TEST_BENCHMARKS, base_run_dir, run_dir,
)

EVAL_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS))
    p.add_argument("--model-path", default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--base", action="store_true", help="Evaluate the untrained base model as step 0.")
    p.add_argument("--runs-root", default=str(ROOT / "runs"))
    p.add_argument("--benchmarks", nargs="+", default=TEST_BENCHMARKS)
    p.add_argument("--steps", default="all", help="'all' saved checkpoints, or a comma list, e.g. 10,20,100.")
    p.add_argument("--k", type=int, default=EVAL_K)
    p.add_argument("--num-problems", type=int, default=EVAL_NUM_PROBLEMS)
    p.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--vllm-gpu-ids", required=True)
    p.add_argument("--vllm-base-port", type=int, default=8100)
    p.add_argument("--vllm-executable", default="vllm")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    p.add_argument("--vllm-max-model-len", type=int, default=41000)
    args = p.parse_args(argv)
    if not args.base and (args.method is None or args.seed is None):
        p.error("--method and --seed are required unless --base is given")
    return args


def load_problems(name: str, num_problems: int):
    """tropic.eval.load_benchmark plus the offline TROPIC_EVAL_DATA_DIR override
    (a local folder with one sub-folder per benchmark name)."""
    from datasets import load_dataset

    from tropic.eval import BENCHMARKS, EvalProblem

    if name not in BENCHMARKS:
        raise SystemExit(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)} (add it to tropic/eval.py)")
    path, split = BENCHMARKS[name]
    local = os.environ.get("TROPIC_EVAL_DATA_DIR")
    if local:
        path = f"{local}/{name}"
    ds = load_dataset(path, split=split)
    ds = ds.select(range(min(num_problems, len(ds))))
    return [EvalProblem(benchmark=name, problem_id=str(r.get("id") or r.get("problem_idx") or i),
                        problem=r["problem"], answer=str(r["answer"])) for i, r in enumerate(ds)]


def answer_entropy(answers: list[str | None]) -> float:
    """Entropy (nats) of the extracted final answers of one problem's k samples;
    a missing answer counts as its own category."""
    counts = Counter(a if a is not None else "<none>" for a in answers)
    n = sum(counts.values())
    return -sum(c / n * math.log(c / n) for c in counts.values()) if n else 0.0


def score_choices(choices: list[dict], gold: str, tokenizer, max_new_tokens: int) -> list[dict]:
    """choices: vLLM completion choices for ONE problem (dicts with text/finish_reason)."""
    from tropic.eval import extract_boxed_answer, grade

    out = []
    for c in choices:
        text = c["text"]
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if tokenizer is not None else None
        ans = extract_boxed_answer(text)
        finish = c.get("finish_reason")
        out.append(dict(
            correct=bool(grade(text, gold)),
            n_tokens=n_tok,
            finish_reason=finish,
            truncated=(finish == "length") if finish is not None else (n_tok is not None and n_tok >= max_new_tokens),
            has_boxed=ans is not None,
            answer=ans,
        ))
    return out


def summarize(problem_rows: list[dict]) -> dict:
    k = len(problem_rows[0]["samples"]) if problem_rows else 0
    n_correct = [sum(s["correct"] for s in r["samples"]) for r in problem_rows]
    return dict(
        num_problems=len(problem_rows), k=k,
        avg_at_k=sum(n_correct) / (len(problem_rows) * k) if problem_rows else 0.0,
        pass_at_k=sum(c > 0 for c in n_correct) / len(problem_rows) if problem_rows else 0.0,
    )


def generate_choices(ports, lora_name, prompts, k, sampling):
    """Like tropic.vllm_rollout_client.generate_eval_batch_parallel, but keeps the
    whole choice dict (finish_reason) instead of only the text."""
    from tropic.vllm_rollout_client import _generate_eval_on_replica

    shard_prompts = [[] for _ in ports]
    shard_idx = [[] for _ in ports]
    for i, pr in enumerate(prompts):
        shard_prompts[i % len(ports)].append(pr)
        shard_idx[i % len(ports)].append(i)
    kw = dict(max_tokens=sampling["max_tokens"], temperature=sampling["temperature"], top_p=sampling["top_p"],
              n=k, top_k=sampling.get("top_k"), min_p=None)
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = {ex.submit(_generate_eval_on_replica, port, lora_name, sp, kw): idx
                for port, sp, idx in zip(ports, shard_prompts, shard_idx) if sp}
        for fut, idx in futs.items():
            choices = sorted(fut.result(), key=lambda c: c["index"])
            for local, orig in enumerate(idx):
                results[orig] = choices[local * k:(local + 1) * k]
    assert all(r is not None for r in results)
    return results


def main(argv=None):
    args = parse_args(argv)
    preset = MODEL_PRESETS[args.model]
    model_name = args.model_path or preset["hf_name"]
    rd = base_run_dir(args.runs_root, args.model) if args.base else run_dir(
        args.runs_root, args.model, args.method, args.seed, args.variant)

    if args.base:
        steps = [0]
    else:
        saved = sorted(int(d.name[4:]) for d in (rd / "ckpt").glob("step*") if (d / "adapter_config.json").exists())
        if args.steps == "all":
            steps = saved
        else:
            steps = sorted(set(int(s) for s in args.steps.split(",")) & set(saved))
        if not steps:
            raise SystemExit(f"no checkpoints to evaluate under {rd}/ckpt (saved: {saved})")

    todo = [(s, b) for s in steps for b in args.benchmarks
            if args.overwrite or not (rd / "eval" / f"step{s}" / f"{b}.json").exists()]
    if not todo:
        print(f"[skip] everything already evaluated for {rd}")
        return 0

    t0 = time.time()

    def tick(msg):
        print(f"[{time.time() - t0:9.1f}s] {msg}", flush=True)

    from transformers import AutoTokenizer

    from tropic.vllm_rollout_client import launch_vllm_replicas, refresh_lora_adapter, shutdown_vllm_replicas

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    problems = {b: load_problems(b, args.num_problems) for b in {b for _, b in todo}}
    prompts = {}
    for b, probs in problems.items():
        prompts[b] = [tokenizer.apply_chat_template([{"role": "user", "content": p.problem + EVAL_SUFFIX}],
                                                    tokenize=False, add_generation_prompt=True, enable_thinking=True)
                      for p in probs]
    sampling = dict(max_tokens=args.max_new_tokens, temperature=preset["eval_temperature"],
                    top_p=preset["eval_top_p"],
                    top_k=preset["eval_top_k"] if preset["eval_top_k"] and preset["eval_top_k"] > 0 else None)

    gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")]
    (rd / "eval").mkdir(parents=True, exist_ok=True)
    procs, ports = launch_vllm_replicas(
        model_name, gpu_ids, args.vllm_base_port, LORA["r"], log_dir=str(rd / "eval"),
        vllm_executable=args.vllm_executable, gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len)
    try:
        for step in sorted({s for s, _ in todo}):
            if args.base:
                lora_name = model_name  # the served base model itself
            else:
                from tropic.vllm_rollout_client import DEFAULT_LORA_NAME

                refresh_lora_adapter(ports, str(rd / "ckpt" / f"step{step}"))
                lora_name = DEFAULT_LORA_NAME
            for b in [b for s, b in todo if s == step]:
                tick(f"step {step} {b}: {len(prompts[b])} problems x {args.k} samples")
                choices = generate_choices(ports, lora_name, prompts[b], args.k, sampling)
                rows = []
                for p, ch in zip(problems[b], choices):
                    samples = score_choices(ch, p.answer, tokenizer, args.max_new_tokens)
                    rows.append(dict(problem_id=p.problem_id, gold=p.answer, samples=samples,
                                     answer_entropy=answer_entropy([s["answer"] for s in samples])))
                summary = summarize(rows)
                d = rd / "eval" / f"step{step}"
                d.mkdir(parents=True, exist_ok=True)
                meta = dict(model=args.model, model_name=model_name, method=args.method or "base",
                            variant=args.variant, seed=args.seed if not args.base else None, step=step,
                            benchmark=b, sampling=sampling, k=args.k, **summary)
                with gzip.open(d / f"{b}.texts.json.gz", "wt", encoding="utf-8") as f:
                    json.dump([[c["text"] for c in ch] for ch in choices], f)
                (d / f"{b}.json").write_text(json.dumps(dict(meta=meta, problems=rows)))
                tick(f"step {step} {b}: avg@{args.k}={summary['avg_at_k']:.4f} pass@{args.k}={summary['pass_at_k']:.4f}")
    finally:
        shutdown_vllm_replicas(procs)
        tick("vLLM replicas shut down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
