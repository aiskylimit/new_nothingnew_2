"""One markdown report of every rebuttal result on disk (phase 1 + 2), small enough for one fetch job.

Reads only result files (summary.json, run_check.json, analysis tables); changes nothing, safe while
runs are going. Writes _fetch/sgl-report/report.md and prints it.
    python phase2_report.py [--out _fetch/sgl-report/report.md]
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path

BENCHMARKS = ["aime24", "aime25", "math500", "amc12"]
# eval dir -> (run number, description); anything else that has a summary.json is listed after these
KNOWN = {
    "rebuttal/results/iwc-gain-l05-r1-qwen-1.5b-s1k": ("1", "1.5B ALERD, s1K-1.1"),
    "rebuttal/results/sft-nll-r1-qwen-1.5b-s1k": ("2", "1.5B SFT, s1K-1.1 (dropped)"),
    "rebuttal/results/zeroshot-r1-qwen-1.5b-s1k": ("3", "1.5B zero-shot (dropped)"),
    "rebuttal/results/iwc-gain-l05-lora-qwen25-7b-s1k": ("4", "Qwen2.5-7B ALERD LoRA, s1K-1.1"),
    "rebuttal/results/iwc-gain-l05-lora-qwen3-8b-s1k": ("7", "Qwen3-8B ALERD LoRA, s1K-1.1"),
    "rebuttal/results/ansstep-mention-r1-qwen-1.5b-s1k": ("10", "1.5B answer-steps, s1K-1.1"),
    "rebuttal/results/sft-asft-r1-qwen-1.5b-palign": ("13", "ASFT, 966 traces"),
    "rebuttal/results/sft-vcore-r1-qwen-1.5b-palign": ("14", "VCORE, 966 traces"),
    "results/positional/runs/r6": ("15", "positional R6"),
    "results/w4/runs/r1q1.5b__alg__P1__S0__tau1_lam0.5_c2__seed42": ("18", "W4 ALERD P1/S0"),
    "results/w4/runs/r1q1.5b__alg__P0__S1__tau1_lam0.5_c2__seed42": ("20", "W4 ALERD P0/S1"),
    "results/hard/runs/r1q1.5b__h59__P0__S0__seed42": ("30", "ALG-hard h59"),
}
SUMMARY_GLOBS = ["rebuttal/results/*/summary.json", "results/*/runs/*/summary.json"]
TABLES = [  # analysis outputs copied in as they are
    "rebuttal/results/comparison-table.md",
    "results/w4/analysis/table_r_w4a.md",
    "results/w4/analysis/table_r_w4b.md",
    "results/hard/analysis/table_r_h1.md",
]
JSONS = [  # small analysis json, shown in a code block
    "results/positional/analysis/a2_summary.json",
    "results/positional/decision.json",
    "results/hard/selection/alg-h59/manifest.json",
]
MAX_JSON_CHARS = 6000


def pct(value):
    return "-" if value is None else f"{100 * value:.2f}"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_table(root: Path) -> list[str]:
    dirs = sorted({p.parent.relative_to(root).as_posix() for g in SUMMARY_GLOBS for p in root.glob(g)})
    order = sorted(dirs, key=lambda d: (d not in KNOWN, int(KNOWN[d][0]) if d in KNOWN else 0, d))
    head = ["Run", "Description"] + [f"{b} p@1" for b in BENCHMARKS] + ["Avg p@1", "Avg p@3", "Len", "Check", "Eval dir"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d in order:
        try:
            rows = {r["benchmark"]: r for r in json.loads((root / d / "summary.json").read_text())}
        except Exception as error:  # noqa: BLE001 - a half-written file must not stop the report
            lines.append(f"| ? | unreadable summary.json ({error}) | " + "| " * (len(head) - 3) + f"{d} |")
            continue
        p1 = [rows[b].get("pass@1", rows[b].get("accuracy")) if b in rows else None for b in BENCHMARKS]
        p3 = [rows[b].get("pass@3") if b in rows else None for b in BENCHMARKS]
        length = mean([r.get("length", r.get("avg_length", r.get("mean_length"))) for r in rows.values()])
        extra = sorted(set(rows) - set(BENCHMARKS))
        check_file = root / d / "run_check.json"
        check = "-"
        if check_file.is_file():
            try:
                c = json.loads(check_file.read_text())
                check = "OK" if c.get("all_ok") else "NOT OK"
            except Exception:  # noqa: BLE001
                check = "?"
        run, desc = KNOWN.get(d, ("", d.split("/")[-1]))
        if extra:
            desc += f" (+{', '.join(extra)})"
        lines.append("| " + " | ".join([run, desc, *map(pct, p1), pct(mean(p1)), pct(mean(p3)),
                                         "-" if length is None else f"{length:.0f}", check, f"`{d}`"]) + " |")
    lines.append("")
    lines.append("Avg = mean over the benchmarks present (pass@k in %). Len = mean generation length if the summary has it.")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="_fetch/sgl-report/report.md")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent

    out = [f"# SGL rebuttal results\n", f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S} on the server.\n",
           "## Runs\n", *run_table(root), ""]

    for rel in TABLES:
        path = root / rel
        out.append(f"## {rel}\n")
        out.append(path.read_text().rstrip() + "\n" if path.is_file() else "_not written yet_\n")

    for rel in JSONS:
        path = root / rel
        out.append(f"## {rel}\n")
        if not path.is_file():
            out.append("_not written yet_\n")
            continue
        text = path.read_text()
        try:
            text = json.dumps(json.loads(text), indent=1)
        except Exception:  # noqa: BLE001
            pass
        if len(text) > MAX_JSON_CHARS:
            text = text[:MAX_JSON_CHARS] + f"\n... ({len(text)} chars, cut)"
        out.append("```json\n" + text + "\n```\n")

    status = subprocess.run(["bash", str(root / "runs_status.sh")], capture_output=True, text=True, timeout=120)
    out.append("## runs_status.sh\n")
    out.append("```\n" + (status.stdout + status.stderr).rstrip() + "\n```\n")

    report = "\n".join(out)
    path = root / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(report)
    print(f"\n[{path.stat().st_size} bytes -> {path}]")


if __name__ == "__main__":
    main()
