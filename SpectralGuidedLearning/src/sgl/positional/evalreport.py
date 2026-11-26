"""Per-sample records, metrics and paired bootstrap from the raw generations of sgl.eval.evaluate.

The headline pass@1 / pass@3 use math_verify, the grader behind every published number compared
against; the "either" columns (math_verify OR oat) are reported next to them, never mixed in.

  report  --raw-dir R --out-dir runs/<id>/eval [--config <sgl config>] [--baseline name=eval_dir ...]
  resolve --avg1 40.76 --avg3 52.03 --results-dir results    print the run tag with those averages
"""

import argparse
import glob
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from sgl.eval.graders.answer_scoring import extract_boxed, normalize_math
from sgl.eval.graders.palign import _oat_label, active_graders, grade_math_verify
from sgl.positional import common as C


def grade_rows(raw_dir: Path):
    oat = "oat_math_grader" in active_graders()
    for name in C.BENCHMARK_ORDER:
        path = raw_dir / f"{name}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing raw generations: {path}")
        for line in path.open():
            row = json.loads(line)
            texts = [g["text"] for g in row["generations"]]
            verified = grade_math_verify(texts, row["gold"])
            for i, (generation, text, label) in enumerate(zip(row["generations"], texts, verified)):
                by_oat = _oat_label(text, row["gold"]) if oat else None
                yield {
                    "benchmark": name, "problem_id": row["id"], "sample_idx": i, "output": text,
                    "n_output_tokens": generation.get("n_tokens"), "finish_reason": generation.get("finish_reason"),
                    "extracted_answer": extract_boxed(text), "gold": row["gold"], "correct_mathverify": int(label),
                    "correct_oat": by_oat, "correct": int(bool(label) or bool(by_oat)),
                    "has_marker_loop": text.count(C.END_MARKER) >= 2,
                }


def by_problem(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        out.setdefault(row["benchmark"], {}).setdefault(row["problem_id"], []).append(row)
    return out


def metrics(rows: list[dict]) -> dict:
    result = {}
    for name, problems in by_problem(rows).items():
        samples = [r for p in problems.values() for r in p]
        tokens = [r["n_output_tokens"] or 0 for r in samples]
        agree = []
        for group in problems.values():
            answers = {normalize_math(r["extracted_answer"]) if r["extracted_answer"] else None for r in group}
            agree.append(len(answers) == 1 and None not in answers)
        item = {
            "pass@1": float(np.mean([r["correct_mathverify"] for r in samples])),
            "pass@3": float(np.mean([any(r["correct_mathverify"] for r in g) for g in problems.values()])),
            "pass@1_either": float(np.mean([r["correct"] for r in samples])),
            "pass@3_either": float(np.mean([any(r["correct"] for r in g) for g in problems.values()])),
            "n_problems": len(problems),
            "truncation_rate": float(np.mean([r["finish_reason"] == "length" for r in samples])),
            "mean_output_tokens": float(np.mean(tokens)), "median_output_tokens": float(np.median(tokens)),
            "sample_agreement": float(np.mean(agree)),
            "extraction_fail_rate": float(np.mean([r["extracted_answer"] is None for r in samples])),
            "marker_loop_rate": float(np.mean([r["has_marker_loop"] for r in samples])),
        }
        result[name] = item
    keys = [k for k in next(iter(result.values())) if k != "n_problems"]
    result["average"] = {k: float(np.mean([m[k] for m in result.values()])) for k in keys}
    result["average"]["n_problems"] = int(sum(m["n_problems"] for n, m in result.items() if n != "average"))
    return result


def paired_bootstrap(run: list[dict], baseline: list[dict], n_boot: int = 10_000, seed: int = 42) -> dict:
    """Delta of the 4-benchmark average, resampling problems inside each benchmark."""
    run, baseline = by_problem(run), by_problem(baseline)
    out = {"n_boot": n_boot, "seed": seed, "grader": "math_verify"}
    scores = {"p1": lambda g: np.mean([r["correct_mathverify"] for r in g]),
              "p3": lambda g: float(any(r["correct_mathverify"] for r in g))}
    for key, score in scores.items():
        rng = np.random.default_rng(seed)
        point, boot = [], np.zeros(n_boot)
        for name, problems in run.items():
            if set(problems) != set(baseline.get(name, {})):
                raise SystemExit(f"paired bootstrap: problem ids of {name} differ between the two runs")
            delta = np.array([score(problems[i]) - score(baseline[name][i]) for i in problems])
            point.append(delta.mean())
            boot += delta[rng.integers(len(delta), size=(n_boot, len(delta)))].mean(axis=1)
        boot /= len(run)
        out[f"delta_avg_{key}"] = float(np.mean(point))
        out[f"delta_avg_{key}_ci95_lo"] = float(np.percentile(boot, 2.5))
        out[f"delta_avg_{key}_ci95_hi"] = float(np.percentile(boot, 97.5))
    return out


def eval_config(name: str) -> dict:
    from sgl import cli

    config = cli.load_family_config(name, "sgl")
    try:
        vllm = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm = None
    versions = {}
    for package in ("math-verify", "latex2sympy2_extended"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {**config["stages"]["eval"]["args"], "vllm_version": vllm, "grader_versions": versions,
            "active_graders": active_graders()}


def resolve(results_dir: str, avg1: float, avg3: float) -> list[str]:
    tags = []
    for path in sorted(glob.glob(f"{results_dir}/*/summary.json")):
        summaries = C.read_json(path)
        if not isinstance(summaries, list) or len(summaries) != 4 or any("pass@3" not in s for s in summaries):
            continue
        p1 = 100 * np.mean([s["pass@1"] for s in summaries])
        p3 = 100 * np.mean([s["pass@3"] for s in summaries])
        if abs(p1 - avg1) < 0.006 and abs(p3 - avg3) < 0.006:
            tags.append(Path(path).parent.name)
    return tags


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("report", "resolve"))
    parser.add_argument("--raw-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--config", help="sgl config whose eval stage produced the generations")
    parser.add_argument("--baseline", action="append", default=[], help="name=<eval dir with generations.jsonl>")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--avg1", type=float)
    parser.add_argument("--avg3", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "resolve":
        tags = resolve(args.results_dir, args.avg1, args.avg3)
        print(tags[0] if tags else "")
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(grade_rows(Path(args.raw_dir)))
    with (out / "generations.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    C.write_json(out / "metrics.json", {**metrics(rows), "provenance": C.provenance()})
    if args.config:
        C.write_json(out / "eval_config.json", {**eval_config(args.config), "provenance": C.provenance()})
    paired = {}
    for entry in args.baseline:
        name, directory = entry.split("=", 1)
        path = Path(directory) / "generations.jsonl"
        if not path.is_file():
            print(f"paired bootstrap against {name} skipped: no {path}")
            continue
        paired[name] = paired_bootstrap(rows, [json.loads(x) for x in path.open()], seed=args.seed)
    if args.baseline:
        C.write_json(out / "paired_vs_baselines.json", {**paired, "provenance": C.provenance()})
    average = metrics(rows)["average"]
    print(f"{out}: avg pass@1 = {average['pass@1']:.2%}, pass@3 = {average['pass@3']:.2%} (math_verify)")


if __name__ == "__main__":
    main()
