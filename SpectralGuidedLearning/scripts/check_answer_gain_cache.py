"""Check the prefix-cached answer gain against the uncached one on a few samples, and estimate the full run.

Exits non-zero when they disagree (pearson or median abs diff), so `check && run` is safe.

    CUDA_VISIBLE_DEVICES=0 python scripts/check_answer_gain_cache.py --model <model dir> \
        --data-path data/qwen25-7b-s1k/train-segmented.jsonl --limit 2 --total-probes 338912
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--min-pearson", type=float, default=0.999)
    parser.add_argument("--max-median-diff", type=float, default=0.05, help="in nats")
    parser.add_argument("--total-probes", type=int, help="probe count of the full run, from its log")
    args = parser.parse_args()

    out = Path(tempfile.mkdtemp(prefix="answer-gain-check-"))
    results = {}
    for name, extra in (("cached", []), ("uncached", ["--no-prefix-cache"])):
        subprocess.run([
            sys.executable, "-m", "sgl.signals.answer_gain", "--model", args.model,
            "--data-path", args.data_path, "--limit", str(args.limit),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--output", str(out / f"{name}.json"), "--meta-output", str(out / f"{name}.meta.json"), *extra,
        ], check=True)
        results[name] = tuple(json.loads((out / f"{name}{suffix}").read_text()) for suffix in (".json", ".meta.json"))

    (new, new_meta), (old, old_meta) = results["cached"], results["uncached"]
    a = [x for rid in old for x in old[rid]]
    b = [x for rid in old for x in new[rid]]
    diffs = sorted(abs(x - y) for x, y in zip(a, b))
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    norm = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5

    print(f"\n=== {len(a)} step gains, outputs in {out}")
    print(f"abs diff: median {diffs[len(diffs) // 2]:.4g}, 99% {diffs[int(len(diffs) * 0.99)]:.4g}, max {diffs[-1]:.4g}")
    print(f"pearson: {cov / norm if norm else float('nan'):.6f}")
    print(f"scoring time: uncached {old_meta['generate_seconds']:.1f}s, cached {new_meta['generate_seconds']:.1f}s "
          f"({old_meta['generate_seconds'] / new_meta['generate_seconds']:.1f}x), rescored {new_meta['n_rescored']}")
    if args.total_probes:
        hours = new_meta["generate_seconds"] / new_meta["n_probes"] * args.total_probes / 3600
        print(f"estimated full run with cache: {hours:.2f} h (a small sample tends to overestimate)")

    pearson = cov / norm if norm else float("nan")
    # written so that a nan fails
    if not (pearson >= args.min_pearson and diffs[len(diffs) // 2] <= args.max_median_diff):
        sys.exit("FAILED: cached gains do not match the uncached ones")
    print("OK: cached gains match")


if __name__ == "__main__":
    main()
