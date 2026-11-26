"""Evaluation records, manifest, end-of-run checklist and the master results table of the W4 runs.

  eval     --run-dir runs/<id> --config <cfg>   raw/*.jsonl -> eval/samples.jsonl + eval/metrics.json
  finalize --run-dir runs/<id> --signals-dir D  manifest.json + run_check.json (the plan's checklist 5.8)
  master   --root results/w4 [--seed42 uniform=<results dir> --seed42 alg=<results dir>]
           -> analysis/results_master.csv (one row per run) + analysis/table_r_w4b.md

Headline pass@1 / pass@3 use math_verify (every published number); `correct` = math_verify OR oat is
stored next to it, as the plan's section 4.1 protocol asks.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.positional import evalreport
from sgl.robustness import common as C


def repetition_loop(text: str) -> bool:
    """The generation ends in a block of words repeated back to back (>= 5 times for blocks under 10
    words, >= 3 times for longer ones)."""
    words = text.split()[-900:]
    for n in range(1, 301):
        times = 5 if n < 10 else 3
        if len(words) < n * times:
            break
        block = words[-n:]
        if all(words[-(i + 1) * n : len(words) - i * n] == block for i in range(1, times)):
            return True
    return False


def sample_rows(raw_dir: Path, eval_seed: int):
    for row in evalreport.grade_rows(raw_dir):
        text = row["output"]
        yield {
            "benchmark": row["benchmark"],
            "problem_id": row["problem_id"],
            "sample_idx": row["sample_idx"],
            "sampling_seed": eval_seed,
            "generation": text,
            "num_tokens": row["n_output_tokens"],
            "finish_reason": row["finish_reason"],
            "extracted_answer": row["extracted_answer"],
            "reference_answer": row["gold"],
            "correct_mathverify": row["correct_mathverify"],
            "correct_oat": row["correct_oat"],
            "correct": row["correct"],
            "truncated": row["finish_reason"] == "length",
            "repetition_loop": repetition_loop(text),
            "emits_marker": C.END_MARKER in text,
            "has_boxed": "\\boxed" in text,
        }


def compute_metrics(samples: list[dict]) -> dict:
    # evalreport.metrics expects its own field names
    legacy = [
        {
            **s,
            "output": s["generation"],
            "n_output_tokens": s["num_tokens"],
            "gold": s["reference_answer"],
            "has_marker_loop": s["generation"].count(C.END_MARKER) >= 2,
        }
        for s in samples
    ]
    result = evalreport.metrics(legacy)
    frame = pd.DataFrame(samples)
    for name, group in frame.groupby("benchmark"):
        both = group.dropna(subset=["correct_oat"])
        result[name] |= {
            "repetition_loop_rate": float(group.repetition_loop.mean()),
            "emits_marker_rate": float(group.emits_marker.mean()),
            "has_boxed_rate": float(group.has_boxed.mean()),
            "n_grader_disagreements": int((both.correct_mathverify != both.correct_oat.astype(int)).sum()),
            "oat_available": len(both) > 0,
        }
    keys = ("repetition_loop_rate", "emits_marker_rate", "has_boxed_rate")
    names = [n for n in result if n != "average"]
    for key in keys:
        result["average"][key] = float(np.mean([result[n][key] for n in names]))
    result["average"]["n_grader_disagreements"] = int(sum(result[n]["n_grader_disagreements"] for n in names))
    return result


def cmd_eval(args) -> None:
    run = Path(args.run_dir)
    samples = list(sample_rows(run / "raw", args.eval_seed))
    out = run / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = compute_metrics(samples)
    vllm = evalreport.eval_config(args.config) if args.config else None
    C.write_json(
        out / "metrics.json", {**metrics, "eval_config": vllm, "n_samples": len(samples), "provenance": C.provenance()}
    )
    avg = metrics["average"]
    print(
        f"{run.name}: avg pass@1 {avg['pass@1']:.2%}, pass@3 {avg['pass@3']:.2%} (math_verify), {len(samples)} samples"
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []


def cmd_finalize(args) -> None:
    run = Path(args.run_dir)
    meta = C.parse_run_id(run.name)
    timing = C.read_json(run / "timing.json") if (run / "timing.json").is_file() else {}
    signals = Path(args.signals_dir) if args.signals_dir else None
    train_config = C.read_json(run / "train/config.json") if (run / "train/config.json").is_file() else None
    C.write_json(
        run / "manifest.json",
        {
            "run_id": run.name,
            **meta,
            "git": C.git_state(),
            "versions": C.versions(),
            "hardware": C.hardware(),
            "seeds": {
                "train (python/numpy/torch/cuda via transformers.set_seed)": meta["seed"],
                "vllm_sampling": args.eval_seed,
            },
            "hashes": C.read_json(args.hashes) if args.hashes and Path(args.hashes).is_file() else None,
            "signals_dir": str(signals) if signals else None,
            "signals_steps_parquet_sha256": C.sha256(signals / "steps.parquet")
            if signals and (signals / "steps.parquet").is_file()
            else None,
            "signals_manifest": C.read_json(signals / "manifest.json") | {"git": None}
            if signals and (signals / "manifest.json").is_file()
            else None,
            "training_file": train_config.get("dataset_file") if train_config else None,
            "training_file_sha256": train_config.get("dataset_sha256") if train_config else None,
            "train_config": train_config,
            "timing": timing,
            "created_at": C.now(),
        },
    )

    steps = read_jsonl(run / "train/train_log.jsonl")
    health = read_jsonl(run / "train/health.jsonl")
    samples = read_jsonl(run / "eval/samples.jsonl")
    metrics = C.read_json(run / "eval/metrics.json") if (run / "eval/metrics.json").is_file() else None
    finite = lambda v: v is None or (isinstance(v, (int, float)) and math.isfinite(v))  # noqa: E731
    changed = [h for h in health if "changed_fraction" in h]
    recomputed = compute_metrics(samples)["average"] if samples else None
    checks = {
        "manifest_complete": (run / "manifest.json").is_file(),
        "step_log_present": bool(steps),
        "step_log_finite": bool(steps)
        and all(finite(x.get(k)) for x in steps for k in ("loss_weighted", "nll_unweighted", "grad_norm")),
        "sum_w_over_Z_close_to_1": bool(steps) and all(abs(x["sum_w_over_Z"] - 1) < 1e-3 for x in steps),
        "params_changed": bool(changed) and changed[-1]["changed_fraction"] >= 0.5,
        "params_changed_fraction": changed[-1]["changed_fraction"] if changed else None,
        "n_health_warnings": len([h for h in health if "warning" in h]),
        "n_samples": len(samples),
        "samples_complete": len(samples) == C.N_EVAL_ROWS,
        "metrics_recomputable": bool(metrics and recomputed)
        and all(abs(metrics["average"][k] - recomputed[k]) < 1e-12 for k in ("pass@1", "pass@3")),
    }
    checks["all_ok"] = all(
        v for k, v in checks.items() if k not in ("params_changed_fraction", "n_health_warnings", "n_samples")
    )
    # informational: with uncommitted changes the manifest carries the diff instead of a clean hash
    checks["git_clean"] = not C.git_state()["dirty"]
    C.write_json(
        run / "run_check.json",
        {
            **checks,
            "note": "commit hash: manifest.json git; results_master.csv "
            "row and backup off the machine are checked by hand",
            "provenance": C.provenance(),
        },
    )
    print(f"{run.name}: checklist {'OK' if checks['all_ok'] else 'NOT OK'} -> {run / 'run_check.json'}")


def run_row(run: Path) -> dict | None:
    metrics_path = run / "eval/metrics.json"
    if not metrics_path.is_file():
        return None
    meta = C.parse_run_id(run.name)
    metrics = C.read_json(metrics_path)
    manifest = C.read_json(run / "manifest.json") if (run / "manifest.json").is_file() else {}
    epochs = read_jsonl(run / "train/epoch_log.jsonl")
    summary = C.read_json(run / "train/train_summary.json") if (run / "train/train_summary.json").is_file() else {}
    row = {
        "run_id": run.name,
        "method": meta["method"],
        "probe": meta["probe"],
        "seg": meta["seg"],
        "tau": C.TEMPERATURE,
        "lambda": C.INTERPOLATION,
        "c": C.CLIP,
        "seed": meta["seed"],
    }
    for name in (*C.BENCHMARK_ORDER, "average"):
        row[f"{name}_p1"] = 100 * metrics[name]["pass@1"]
        row[f"{name}_p3"] = 100 * metrics[name]["pass@3"]
    avg = metrics["average"]
    row |= {
        "truncation_rate": avg["truncation_rate"],
        "sample_agreement": avg["sample_agreement"],
        "repetition_loop_rate": avg.get("repetition_loop_rate"),
        "emits_marker_rate": avg.get("emits_marker_rate"),
    }
    last = epochs[-1] if epochs else {}
    row |= {
        "final_nll": last.get("dataset_nll_all", summary.get("final_nll_unweighted")),
        "final_nll_prefix": last.get("dataset_nll_prefix"),
        "final_nll_continuation": last.get("dataset_nll_continuation"),
    }
    signals = manifest.get("signals_dir")
    scoring = None
    if signals and Path(signals, "summary.json").is_file():
        scoring = C.read_json(Path(signals, "summary.json")).get("scoring_seconds")
    timing = manifest.get("timing") or {}
    row |= {
        "scoring_seconds": scoring,
        "train_hours": summary.get("wall_clock_hours"),
        "eval_seconds": timing.get("eval_seconds"),
        "signals_dir": signals,
        "commit": (manifest.get("git") or {}).get("commit"),
    }
    return row


def seed42_row(name: str, directory: Path) -> dict | None:
    """Seed 42 of uniform / ALG default = the existing E0 / E7 run (results/<tag>/summary.json)."""
    path = directory / "summary.json"
    if not path.is_file():
        print(f"seed 42 {name}: no {path}")
        return None
    by_name = {s["benchmark"]: s for s in C.read_json(path) if isinstance(s, dict)}
    probe, seg = ("none", "none") if name == "uniform" else ("P0", "S0")
    row = {
        "run_id": f"{C.run_id(name, probe, seg, 42)} (= {directory.name})",
        "method": name,
        "probe": probe,
        "seg": seg,
        "tau": C.TEMPERATURE,
        "lambda": C.INTERPOLATION,
        "c": C.CLIP,
        "seed": 42,
    }
    for bench in C.BENCHMARK_ORDER:
        if bench not in by_name:
            return None
        row[f"{bench}_p1"], row[f"{bench}_p3"] = 100 * by_name[bench]["pass@1"], 100 * by_name[bench]["pass@3"]
    row["average_p1"] = float(np.mean([row[f"{b}_p1"] for b in C.BENCHMARK_ORDER]))
    row["average_p3"] = float(np.mean([row[f"{b}_p3"] for b in C.BENCHMARK_ORDER]))
    return row


def cmd_master(args) -> None:
    root = Path(args.root)
    rows = [r for r in (run_row(p) for p in sorted((root / "runs").glob("*__seed*"))) if r]
    for entry in args.seed42:
        name, directory = entry.split("=", 1)
        row = seed42_row(name, Path(directory))
        if row:
            rows.append(row)
    frame = pd.DataFrame(rows)
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "analysis/results_master.csv", index=False)
    if frame.empty:
        print("no finished run yet")
        return

    benches = (*C.BENCHMARK_ORDER, "average")
    cell = lambda p1, p3: f"{p1:.2f} / {p3:.2f}"  # noqa: E731
    lines = [
        "| Method | AIME25 p@1/p@3 | AIME24 p@1/p@3 | AMC12 p@1/p@3 | MATH500 p@1/p@3 | Avg p@1/p@3 | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, numbers in C.PUBLISHED.items():
        key = {"aime25": "aime25", "aime24": "aime24", "amc12": "amc12", "math500": "math500", "average": "avg"}
        lines.append(f"| {label} (paper) | " + " | ".join(cell(*numbers[key[b]]) for b in benches) + " | 1 |")
    for method, probe, seg, label in (
        ("uniform", "none", "none", "Uniform"),
        ("alg", "P0", "S0", "ALG default (P0, S0)"),
    ):
        group = frame[(frame.method == method) & (frame.probe == probe) & (frame.seg == seg)]
        if len(group) >= 2:

            def fmt(b, group=group):
                return (
                    f"{group[f'{b}_p1'].mean():.2f}±{group[f'{b}_p1'].std():.2f} / "
                    f"{group[f'{b}_p3'].mean():.2f}±{group[f'{b}_p3'].std():.2f}"
                )

            lines.append(
                f"| {label} mean ± std (seeds {sorted(group.seed.tolist())}) | "
                + " | ".join(fmt(b) for b in benches)
                + f" | {len(group)} |"
            )
    variants = frame[(frame.method == "alg") & ~((frame.probe == "P0") & (frame.seg == "S0"))]
    for _, r in variants.sort_values(["seg", "probe", "seed"]).iterrows():
        label = f"ALG, {r.probe}" if r.seg == "S0" else f"ALG, {r.seg}"
        lines.append(
            f"| {label} (seed {r.seed}) | " + " | ".join(cell(r[f"{b}_p1"], r[f"{b}_p3"]) for b in benches) + " | 1 |"
        )
    lo, _ = C.UNIFORM_AVG
    below = variants[variants.average_p1 < lo]
    lines += [
        "",
        f"Uniform reference for the decision (plan section 0): {C.UNIFORM_AVG[0]} / {C.UNIFORM_AVG[1]}. "
        f"Variants below it on avg pass@1: {', '.join(below.run_id) if len(below) else 'none'}.",
    ]
    (root / "analysis/table_r_w4b.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_hashes(args) -> None:
    """SHA-256 of the trace file, the tokenizer files and the base model weights (computed once)."""
    out = Path(args.output)
    if out.is_file():
        print(f"{out} exists")
        return
    model = Path(args.model_dir)
    files = {"traces": Path(args.traces)}
    for pattern in ("tokenizer*.json", "*.safetensors", "config.json"):
        for path in sorted(model.glob(pattern)):
            files[f"model/{path.name}"] = path
    C.write_json(
        out,
        {name: C.sha256(path) for name, path in files.items() if path.is_file()}
        | {"model_dir": str(model), "created_at": C.now()},
    )
    print(f"hashes of {len(files)} files -> {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    hashes = sub.add_parser("hashes")
    hashes.add_argument("--traces", required=True)
    hashes.add_argument("--model-dir", required=True)
    hashes.add_argument("--output", required=True)
    ev = sub.add_parser("eval")
    ev.add_argument("--run-dir", required=True)
    ev.add_argument("--config", help="sgl config of the eval stage (recorded in metrics.json)")
    ev.add_argument("--eval-seed", type=int, default=42)
    fin = sub.add_parser("finalize")
    fin.add_argument("--run-dir", required=True)
    fin.add_argument("--signals-dir")
    fin.add_argument("--hashes")
    fin.add_argument("--eval-seed", type=int, default=42)
    master = sub.add_parser("master")
    master.add_argument("--root", default="results/w4")
    master.add_argument("--seed42", action="append", default=[], help="uniform=<results/tag> or alg=<results/tag>")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {"hashes": cmd_hashes, "eval": cmd_eval, "finalize": cmd_finalize, "master": cmd_master}[args.action](args)


if __name__ == "__main__":
    main()
