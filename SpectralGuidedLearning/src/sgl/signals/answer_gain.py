"""Per-step answer-information gain: how much each reasoning step raises the frozen student's
log-probability of the gold final answer.

    gain_k = log p(answer | prompt, steps_1..k) - log p(answer | prompt, steps_1..k-1)

p(answer | context) is read by appending a fixed probe ("... the final answer is \\boxed{") and
scoring the gold answer tokens + "}" with vLLM prompt_logprobs. All probes of a sample share the
response prefix, so with vLLM's prefix cache each probe only computes what the earlier probes have
not. vLLM skips reading that cache by default whenever prompt_logprobs is requested, so it is turned
back on here; the returned rows then cover only the computed tail of the prompt. The gold answer
is the last \\boxed{...} of the training response (P-ALIGN keeps only answer-verified samples).
Output: {id: [gain per step]} JSON, aligned with train-segmented.jsonl steps.
"""

import argparse
import json
import time
from pathlib import Path

from sgl.eval.graders.answer_scoring import extract_boxed

PROBE = "\n\nTherefore, the final answer is \\boxed{"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--logp-output", help="also save log p(answer) before step 1 and after every step")
    parser.add_argument("--probe", default=PROBE, help="text appended before the gold answer (default: P0)")
    parser.add_argument(
        "--meta-output",
        help="also save the probe, its token ids, per-trace probe context lengths and skipped steps, timing",
    )
    parser.add_argument(
        "--no-prefix-cache", action="store_true",
        help="recompute every probe from the first token (very slow; the reference to compare against)",
    )
    return parser


def answer_logp(rows, ids: list[int], n_answer: int) -> float | None:
    """Sum of the log-probabilities of the last n_answer prompt tokens.

    Read from the end: after a prefix-cache hit `rows` has no entries for the cached positions.
    None when it does not even reach back to the first answer token.
    """
    if len(rows) < n_answer:
        return None
    # each row is keyed by its own token id, so a misaligned read raises instead of scoring wrongly
    return sum(rows[i][ids[i]].logprob for i in range(-n_answer, 0))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(line) for line in open(args.data_path)][: args.limit]
    started = time.time()
    probe_ids = tokenizer(args.probe, add_special_tokens=False)["input_ids"]
    longest, too_long = {}, {}  # per trace: longest probe context, steps whose probe did not fit

    prompts, index = [], []  # index[i] = (record id, step k (0 = before any step), n answer tokens)
    skipped = 0
    for record in records:
        answer = extract_boxed(record["response"])
        if not answer:
            skipped += 1
            continue
        answer_ids = tokenizer(answer + "}", add_special_tokens=False)["input_ids"]
        start = record["response_token_span"][0]
        cuts = [start] + [step["token_end"] for step in record["steps"]]
        for k, cut in enumerate(cuts):
            ids = record["input_ids"][:cut] + probe_ids + answer_ids
            longest[record["id"]] = max(longest.get(record["id"], 0), len(ids))
            if len(ids) >= args.max_model_len:
                too_long.setdefault(record["id"], []).append(k)
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            index.append((record["id"], k, len(answer_ids)))
    print(f"{len(prompts):,} probes over {len(records) - skipped} samples (skipped {skipped} without \\boxed)")

    llm = LLM(
        model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enable_prefix_caching=True, enforce_eager=True,
    )

    def score(batch, use_cache):
        params = SamplingParams(max_tokens=1, prompt_logprobs=0, skip_reading_prefix_cache=not use_cache)
        outputs = llm.generate([prompts[i] for i in batch], params)
        return [
            answer_logp(out.prompt_logprobs, prompts[i]["prompt_token_ids"], index[i][2])
            for i, out in zip(batch, outputs)
        ]

    generate_started = time.time()
    totals = score(range(len(prompts)), not args.no_prefix_cache)
    # identical probes (two steps ending at the same token) can be cached past the answer start
    retry = [i for i, total in enumerate(totals) if total is None]
    if retry:
        print(f"rescoring {len(retry):,} probes without the prefix cache")
        for i, total in zip(retry, score(retry, False)):
            totals[i] = total
    generate_seconds = time.time() - generate_started
    print(f"scored {len(prompts):,} probes in {generate_seconds:.1f}s")

    logp = {(rid, k): total for (rid, k, _), total in zip(index, totals)}  # log p(answer | steps 1..k)

    gains, logps = {}, {}
    for record in records:
        rid, n_steps = record["id"], len(record["steps"])
        if (rid, 0) not in logp:
            continue
        values = [logp.get((rid, k)) for k in range(n_steps + 1)]
        # a step whose probe was skipped (too long) inherits the last known value: zero gain
        for k in range(1, len(values)):
            if values[k] is None:
                values[k] = values[k - 1]
        logps[rid] = values
        gains[rid] = [values[k] - values[k - 1] for k in range(1, n_steps + 1)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(gains))
    if args.logp_output:
        Path(args.logp_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.logp_output).write_text(json.dumps(logps))
    if args.meta_output:
        Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_output).write_text(json.dumps({
            "probe": args.probe, "probe_token_ids": probe_ids,
            "probe_tokens": tokenizer.convert_ids_to_tokens(probe_ids),
            "model": args.model, "data_path": args.data_path, "max_model_len": args.max_model_len,
            "n_probes": len(prompts), "n_samples_scored": len(gains), "n_samples_without_boxed": skipped,
            "max_probe_context_len": {str(k): v for k, v in longest.items()},
            "steps_exceeding_context": {str(k): v for k, v in too_long.items()},
            "seconds": time.time() - started, "generate_seconds": generate_seconds,
            "prefix_cache": not args.no_prefix_cache, "n_rescored": len(retry),
        }, indent=1))
    print(f"wrote gains for {len(gains)} samples -> {args.output}")


if __name__ == "__main__":
    main()
