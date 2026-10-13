#!/usr/bin/env python3
"""vLLM eval worker: one model on the GPU given by CUDA_VISIBLE_DEVICES.

--role generate: greedy answers of a policy model for prepared items.
    mode "mc"  -> one token restricted to the option letters (no free-text parsing),
    mode "gen" -> greedy generation up to the item's max_tokens.
--role guard: Qwen3Guard-Gen response moderation of (prompt, response) pairs,
    same chat formatting as the training reward (grit.data.format_safety_conversation).
Output is appended per batch, so an interrupted run resumes where it stopped.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LETTERS = "ABCDEFGHIJKLMNOP"
READY_MARKER = "[EVAL_WORKER_READY]"
T0 = time.time()


def tick(label):
    print(f"[{time.time() - T0:8.1f}s] {label}", flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def done_uids(path):
    done = set()
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["uid"])
            except (ValueError, KeyError):
                pass  # a truncated last line from a killed run is regenerated
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["generate", "guard"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--items", required=True, help="generate: prepared items; guard: generations to judge")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--guard-max-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=512, help="Concurrent sequences inside vLLM")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    args = parser.parse_args()

    items = read_jsonl(args.items)
    done = done_uids(args.out)
    pending = [item for item in items if item["uid"] not in done]
    tick(f"[{args.label}/{args.role}] {len(done)} done, {len(pending)} pending")
    if not pending:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tokenizer=args.model, dtype=args.dtype, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, seed=0,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens)
    # The orchestrator holds a per-GPU startup lock until this line, so concurrent
    # engines never profile memory at the same time.
    print(READY_MARKER, flush=True)
    letter_ids = []
    if args.role == "generate":
        for letter in LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter!r} is not a single token: {ids}")
            letter_ids.append(ids[0])
        # MC items are cheap; long generations batch better together.
        pending.sort(key=lambda item: (item["mode"] != "mc", item.get("max_tokens", 1)))
    else:
        from grit.data import format_safety_conversation

    started, finished = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts, params, keep = [], [], []
            for item in batch:
                if args.role == "generate":
                    # raw_prompt: the benchmark's official prompt is already a full chat string (LiveCodeBench).
                    text = item["prompt"] if item.get("raw_prompt") else tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)
                    n_prompt = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    if item["mode"] == "mc":
                        sampling = SamplingParams(temperature=0.0, max_tokens=1,
                                                  allowed_token_ids=letter_ids[: item["n_options"]])
                    else:
                        budget = min(item["max_tokens"], args.max_model_len - n_prompt)
                        if budget < 16:
                            handle.write(json.dumps({"uid": item["uid"], "response": "", "choice": None,
                                                     "finish_reason": "prompt_too_long", "n_tokens": 0}) + "\n")
                            continue
                        sampling = SamplingParams(temperature=0.0, max_tokens=budget)
                else:
                    text = format_safety_conversation(tokenizer, item["prompt"], item["response"])
                    n_prompt = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                    if n_prompt + args.guard_max_tokens > args.max_model_len:
                        # Keep the head of the conversation: the label depends on the prompt and response start.
                        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                        text = tokenizer.decode(ids[: args.max_model_len - args.guard_max_tokens])
                    sampling = SamplingParams(temperature=0.0, max_tokens=args.guard_max_tokens)
                prompts.append(text)
                params.append(sampling)
                keep.append(item)
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False) if prompts else []
            for item, output in zip(keep, outputs, strict=True):
                completion = output.outputs[0]
                text = completion.text.strip()
                if args.role == "generate":
                    record = {"uid": item["uid"], "response": text,
                              "choice": text[:1] if item["mode"] == "mc" else None,
                              "finish_reason": completion.finish_reason, "n_tokens": len(completion.token_ids)}
                else:
                    record = {"uid": item["uid"], "guard_output": text}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished += len(batch)
            rate = finished / max(1e-3, time.time() - started)
            tick(f"[{args.label}/{args.role}] {len(done) + finished}/{len(items)} | {rate:.1f} item/s")
    tick(f"[{args.label}/{args.role}] finished -> {args.out}")


if __name__ == "__main__":
    main()
