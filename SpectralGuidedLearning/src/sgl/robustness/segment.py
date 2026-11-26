"""Re-segment train-segmented.jsonl (S0, sentence rule) into another step segmentation.

Only "steps" changes: input_ids, the response span and the stop token stay byte-identical, so every
variant scores and trains on the same tokens. Text rules are mapped to tokens exactly as
sgl.data.prepare does (one tokenizer pass, a token belongs to the step holding its first character);
chunks are fixed blocks of response tokens (the last one shorter).

Writes <output> and <output>.check.json (token coverage, stop token outside the steps, steps per trace).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sgl.robustness import common as C
from sgl.segment.sentence import encode_with_offsets, step_token_spans


def chunk_spans(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(first, min(first + size, end)) for first in range(start, end, size)]


def resegment(record: dict, seg: str, tokenizer) -> list[tuple[int, int]]:
    kind, parameter = C.SEGMENTATIONS[seg]
    start, end = record["response_token_span"]
    if kind == "chunk":
        return chunk_spans(start, end, parameter)
    ids, token_starts = encode_with_offsets(tokenizer, record["response"])
    if ids != record["input_ids"][start:end]:
        raise ValueError(f"record {record['id']}: the tokenizer does not reproduce the stored response ids")
    return step_token_spans(record["response"], token_starts, offset=start, segmenter=parameter)


def coverage_errors(record: dict, spans: list[tuple[int, int]]) -> list[str]:
    start, end = record["response_token_span"]
    errors = []
    if not spans or spans[0][0] != start or spans[-1][1] != end:
        errors.append("steps do not start/end at the response span")
    if any(b[0] != a[1] for a, b in zip(spans, spans[1:])):
        errors.append("steps are not contiguous")
    if any(last <= first for first, last in spans):
        errors.append("empty step")
    if sum(last - first for first, last in spans) != end - start:
        errors.append("sum of step lengths != response tokens")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl (S0)")
    parser.add_argument("--seg", required=True, choices=sorted(C.SEGMENTATIONS))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = [json.loads(line) for line in open(args.data_path)]
    counts, errors, s0_mismatch = [], {}, 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            spans = resegment(record, args.seg, tokenizer)
            problems = coverage_errors(record, spans)
            if problems:
                errors[str(record["id"])] = problems
            if args.seg == "S0":
                s0_mismatch += spans != [(s["token_start"], s["token_end"]) for s in record["steps"]]
            out = {**record, "steps": [{"token_start": a, "token_end": b} for a, b in spans], "segmentation": args.seg}
            counts.append(len(spans))
            handle.write(json.dumps(out) + "\n")
    counts = np.array(counts)
    kind, parameter = C.SEGMENTATIONS[args.seg]
    stop_outside = all(len(r["input_ids"]) > r["response_token_span"][1] for r in records)
    report = {
        "seg": args.seg,
        "kind": kind,
        "parameter": parameter,
        "function": "sgl.robustness.segment.chunk_spans"
        if kind == "chunk"
        else f"sgl.segment.SEGMENTERS[{parameter!r}] + sgl.segment.sentence.step_token_spans",
        "n_traces": len(records),
        "n_steps": int(counts.sum()),
        "steps_per_trace": {
            "min": int(counts.min()),
            "p25": float(np.percentile(counts, 25)),
            "median": float(np.median(counts)),
            "p75": float(np.percentile(counts, 75)),
            "max": int(counts.max()),
        },
        "n_single_step_traces": int((counts == 1).sum()),
        "n_response_tokens": int(sum(r["response_token_span"][1] - r["response_token_span"][0] for r in records)),
        "stop_token_outside_steps": stop_outside,
        "n_traces_with_coverage_errors": len(errors),
        "coverage_errors": dict(list(errors.items())[:20]),
        "s0_regression_mismatched_traces": s0_mismatch if args.seg == "S0" else None,
        "provenance": C.provenance({"data": args.data_path}),
    }
    C.write_json(f"{output}.check.json", report)
    print(
        f"{args.seg}: {report['n_steps']:,} steps over {len(records)} traces "
        f"(median {report['steps_per_trace']['median']:.0f}/trace, {report['n_single_step_traces']} single-step) "
        f"-> {output}"
    )
    if errors or not stop_outside or s0_mismatch:
        raise SystemExit(f"segmentation check FAILED, see {output}.check.json")


if __name__ == "__main__":
    main()
