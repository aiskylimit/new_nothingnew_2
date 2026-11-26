"""Summarize the fixed Qwen2.5-7B tau=1, batch-8 tuning grid."""

import argparse
import csv
import json
import re
from pathlib import Path


BENCHMARKS = ("aime24", "aime25", "amc12", "math500")
ARM_RE = re.compile(r"iwc-gain-l(?P<lambda>\d+)-t1-c(?P<clip>\d+)-b8-lora")


def pct(row: dict, key: str) -> str:
    value = row.get(key)
    return f"{value:.2%}" if isinstance(value, (int, float)) else "-"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/qwen25-tuning-tau1-b8")
    parser.add_argument("--track", default="qwen25-7b-palign")
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, markdown = [], ["# Qwen2.5-7B ALG tuning — tau=1, batch=8", "",
                          "Each benchmark cell is `pass@1 / pass@3`.", "",
                          "| Arm | lambda | clip | AIME24 | AIME25 | AMC12 | MATH500 | Avg p@1 | Avg p@3 | Status |",
                          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in args.arms:
        match = ARM_RE.fullmatch(arm)
        if not match:
            raise ValueError(f"unrecognized tuning arm: {arm}")
        path = Path(args.results_dir) / f"{arm}-{args.track}" / "summary.json"
        row = {"arm": arm, "lambda": int(match["lambda"]) / 100, "tau": 1.0,
               "clip": int(match["clip"]), "status": "missing"}
        scores = {}
        if path.exists():
            scores = {item["benchmark"]: item for item in json.loads(path.read_text())}
            if all(name in scores for name in BENCHMARKS):
                row["status"] = "complete"
                row["avg_pass@1"] = sum(scores[name]["pass@1"] for name in BENCHMARKS) / len(BENCHMARKS)
                row["avg_pass@3"] = sum(scores[name]["pass@3"] for name in BENCHMARKS) / len(BENCHMARKS)
                for name in BENCHMARKS:
                    row[f"{name}_pass@1"] = scores[name]["pass@1"]
                    row[f"{name}_pass@3"] = scores[name]["pass@3"]
        rows.append(row)
        benchmark_scores = [
            f"{row[f'{name}_pass@1']:.2%} / {row[f'{name}_pass@3']:.2%}" if name in scores else "-"
            for name in BENCHMARKS
        ]
        markdown.append(
            f"| {arm} | {row['lambda']:.2f} | {row['clip']} | " + " | ".join(benchmark_scores)
            + f" | {pct(row, 'avg_pass@1')} | {pct(row, 'avg_pass@3')} | {row['status']} |"
        )
    fields = ["arm", "lambda", "tau", "clip", "aime24_pass@1", "aime25_pass@1", "amc12_pass@1", "math500_pass@1",
              "aime24_pass@3", "aime25_pass@3", "amc12_pass@3", "math500_pass@3", "avg_pass@1", "avg_pass@3", "status"]
    with (out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = "\n".join(markdown) + "\n"
    (out_dir / "summary.md").write_text(report)
    print(report, end="")
    print(f"tuning report -> {out_dir / 'summary.md'} (+ summary.csv)")


if __name__ == "__main__":
    main()
