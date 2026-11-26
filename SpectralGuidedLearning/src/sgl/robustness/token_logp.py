"""Frozen-student log q0(y_t | x, y_<t) of every response token, saved once per trace.

The `predictability` column of every segmentation (mean token log-prob of a step) is read from this
one file, so the five segmentations need a single forward pass per trace. Same computation as
sgl.signals.predictability (chunked vocabulary projection), output: npz {trace id: float32 array}.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help=".npz")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM

    from sgl.signals.gradients import shift_for_causal_lm
    from sgl.signals.predictability import token_log_probabilities

    started = time.time()
    records = [json.loads(line) for line in open(args.data_path)][: args.limit]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(args.device).eval()
    unembedding = model.get_output_embeddings().weight.float()
    values = {}
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            start, end = record["response_token_span"]
            input_ids = torch.tensor([record["input_ids"]], device=args.device)
            hidden = model.model(input_ids=input_ids).last_hidden_state[0]
            rows, targets = shift_for_causal_lm(hidden, input_ids[0], (start, end))
            values[str(record["id"])] = (
                token_log_probabilities(rows, targets, unembedding, args.chunk_size).float().cpu().numpy()
            )
            if index % 100 == 0:
                print(f"{index}/{len(records)} traces")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    Path(f"{args.output}.meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "data_path": args.data_path,
                "n_traces": len(values),
                "seconds": time.time() - started,
                "peak_gpu_mem_gb": peak,
            },
            indent=1,
        )
    )
    print(f"token log-probs of {len(values)} traces -> {args.output}")


if __name__ == "__main__":
    main()
