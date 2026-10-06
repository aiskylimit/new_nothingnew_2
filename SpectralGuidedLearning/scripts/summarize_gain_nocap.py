"""Markdown summary of the no-capture answer-gain runs, next to the capture-based iwc-gain-l05 rows.

    python scripts/summarize_gain_nocap.py --results-dir results_b200 --output results_b200/summary-iwc-gain-nocap.md
"""

import argparse
import json
from pathlib import Path

BENCHMARKS = ["math500", "aime24", "aime25", "amc12"]
# (label, run tag, results dir override); the reference rows come from the tracked results/ dir
RUNS = [
    ("qwen25-7b, no capture, batch 8", "iwc-gain-nocap-l05-lora-qwen25-7b-palign", None),
    ("qwen25-7b, capture (reference)", "iwc-gain-l05-lora-qwen25-7b-palign", "results"),
    ("qwen3-8b, no capture, batch 32", "iwc-gain-nocap-l05-lora-qwen3-8b-palign", None),
    ("qwen3-8b, capture (reference)", "iwc-gain-l05-lora-qwen3-8b-palign", "results"),
]


def load(directory: Path, tag: str) -> dict[str, dict] | None:
    path = directory / tag / "summary.json"
    if not path.exists():
        return None
    return {row["benchmark"]: row for row in json.loads(path.read_text())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_b200")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lines = ["# Answer-gain IWC (lambda 0.5, no gate): no-capture runs", ""]
    for metric, title in (("pass@1", "pass@1 (%)"), ("pass@3", "pass@3 (%)"), ("truncation_rate", "truncation rate (%)")):
        lines += [f"## {title}", "", "| run | " + " | ".join(BENCHMARKS) + " | mean |", "|---|" + "---|" * (len(BENCHMARKS) + 1)]
        for label, tag, override in RUNS:
            rows = load(Path(override or args.results_dir), tag)
            if rows is None:
                lines.append(f"| {label} | " + " | ".join(["n/a"] * (len(BENCHMARKS) + 1)) + " |")
                continue
            values = [100 * rows[b][metric] if b in rows and metric in rows[b] else None for b in BENCHMARKS]
            seen = [v for v in values if v is not None]
            cells = [f"{v:.1f}" if v is not None else "n/a" for v in values]
            mean = f"{sum(seen) / len(seen):.1f}" if seen else "n/a"
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
        lines.append("")
    lines += [
        "No-capture runs skip the gradient capture and per-sample SVD; at p=1.0 the weights are the same as the",
        "capture-based arm, up to floating-point ties. Reference rows are read from `results/` (other batch and",
        "hardware); n/a means the run has no summary.json.",
        "",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
