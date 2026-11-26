"""End of one ALG-hard run: manifest.json, eval/paired_vs_baselines.json and run_check.json."""

import argparse
import json
import math
from pathlib import Path

from sgl.hardsel import common as C
from sgl.positional import evalreport
from sgl.robustness.runs import compute_metrics, read_jsonl

GRADERS = {"either": "correct", "mathverify": "correct_mathverify"}


def graded_rows(raw_dir: Path) -> list[dict]:
    return list(evalreport.grade_rows(raw_dir))


def paired(run: list[dict], baseline: list[dict]) -> dict:
    """Paired bootstrap of the 4-benchmark average, with the headline grader (math_verify OR oat) and
    with math_verify alone. Delta = run - baseline, in fractions."""
    out = {}
    for name, column in GRADERS.items():
        remap = lambda rows, column=column: [{**r, "correct_mathverify": r[column]} for r in rows]  # noqa: E731
        out[name] = evalreport.paired_bootstrap(remap(run), remap(baseline))
        out[name]["grader"] = name
    return out


def write_paired(run: Path, baselines: dict[str, Path]) -> dict:
    if not (run / "raw").is_dir():
        return {}
    rows, out = graded_rows(run / "raw"), {}
    for name, raw in baselines.items():
        if not all((raw / f"{b}.jsonl").is_file() for b in C.BENCHMARK_ORDER):
            out[name] = {"missing": str(raw)}
            continue
        out[name] = paired(rows, graded_rows(raw))
    C.write_json(run / "eval/paired_vs_baselines.json", {**out, "provenance": C.provenance()})
    return out


def finalize(args) -> None:
    run = Path(args.run_dir)
    meta = C.parse_run_id(run.name)
    selection = Path(args.selection_root) / meta["arm"]
    load = lambda p: C.read_json(p) if Path(p).is_file() else None  # noqa: E731
    train_config, timing = load(run / "train/config.json"), load(run / "timing.json") or {}
    stats, selected = load(selection / "selection_stats.json"), load(selection / "manifest.json")
    C.write_json(run / "manifest.json", {
        "run_id": run.name, **meta,
        "label": selected.get("label") if selected else None,
        "git": C.git_state(), "versions": C.versions(), "hardware": C.hardware(),
        "seeds": {"train (python/numpy/torch/cuda)": meta["seed"], "vllm_sampling": args.eval_seed},
        "hashes": load(args.hashes) if args.hashes else None,
        "training_file": train_config.get("dataset_file") if train_config else None,
        "training_file_sha256": train_config.get("dataset_sha256") if train_config else None,
        "selection_manifest": {**selected, "git": None} if selected else None,
        "train_config": train_config, "timing": timing, "created_at": C.now(),
    })

    steps = read_jsonl(run / "train/train_log.jsonl")
    health = read_jsonl(run / "train/health.jsonl")
    epochs = read_jsonl(run / "train/epoch_log.jsonl")
    samples = read_jsonl(run / "eval/samples.jsonl")
    metrics = load(run / "eval/metrics.json")
    data_check = load(run / "train/data_check.json") or {}

    expected_steps = None
    if train_config and data_check.get("n_traces"):
        hyper = train_config["hyperparameters"]
        expected_steps = math.ceil(data_check["n_traces"] / hyper["effective_batch"]) * int(hyper["epochs"])
    finite = lambda v: v is None or (isinstance(v, (int, float)) and math.isfinite(v))  # noqa: E731
    changed = [h for h in health if "changed_fraction" in h]
    total = sum(x["n_tok_total"] for x in steps)
    kept_frac = sum(x["n_tok_kept"] for x in steps) / total if total else None
    recomputed = compute_metrics(samples)["average"] if samples else None
    checks = {
        "manifest_complete": bool(train_config and selected),
        "step_log_finite": bool(steps) and all(
            finite(x.get(k)) for x in steps for k in ("loss", "nll_kept", "nll_all_unweighted", "grad_norm")),
        "n_optimizer_steps": len(steps),
        "optimizer_steps_as_expected": len(steps) == expected_steps,
        "params_changed_at_step_64": bool(changed) and changed[-1]["changed_fraction"] >= 0.5,
        "kept_frac_in_training": kept_frac,
        "kept_frac_matches_selection": bool(stats and kept_frac is not None
                                            and abs(kept_frac - stats["kept_token_frac"]) < 5e-3),
        "sum_w_over_Z_close_to_1": bool(steps) and all(abs(x["sum_w_over_Z"] - 1) < 1e-3 for x in steps),
        "n_health_warnings": len([h for h in health if "warning" in h]),
        "n_samples": len(samples),
        "samples_complete": len(samples) == args.n_samples,
        "metrics_recomputable": bool(metrics and recomputed) and all(
            abs(metrics["average"][k] - recomputed[k]) < 1e-12 for k in ("pass@1", "pass@3")),
        "epoch_log_0_to_last": bool(train_config) and [x["epoch"] for x in epochs]
        == list(range(int(train_config["hyperparameters"]["epochs"]) + 1)),
    }
    informational = ("n_optimizer_steps", "kept_frac_in_training", "n_health_warnings", "n_samples")
    checks["all_ok"] = all(v for k, v in checks.items() if k not in informational)
    checks["git_clean"] = not C.git_state()["dirty"]  # with local changes the manifest carries the diff
    C.write_json(run / "run_check.json", {**checks, "expected_optimizer_steps": expected_steps,
                                          "provenance": C.provenance()})
    failed = [k for k, v in checks.items() if v is False and k not in ("all_ok", "git_clean")]
    print(f"{run.name}: checklist {'OK' if checks['all_ok'] else 'NOT OK ' + json.dumps(failed)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection-root", default="results/hard/selection")
    parser.add_argument("--hashes")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=(30 + 30 + 83 + 500) * 3)
    return parser


def main(argv: list[str] | None = None) -> None:
    finalize(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
