"""Everything of results/hard/analysis/, rebuilt from the files on disk.

  rule     write the pre-registered decision rule to decision.json (before any training)
  report   paired bootstraps, Tables R-H1 / R-H2, Figures H-1 / H-2, results_master.csv, decision.json
  needs-c  exit 0 if decision.json asks for group C (H59 seeds 43, 44), 1 otherwise

The headline grader is `correct` = math_verify OR oat (the plan's Table 2 protocol); math_verify alone is
reported next to it.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.hardsel import common as C
from sgl.hardsel.runs import GRADERS, graded_rows, paired, write_paired
from sgl.positional import evalreport
from sgl.robustness import common as W4
from sgl.robustness.runs import read_jsonl

BENCH_LABELS = {"aime25": "AIME25", "aime24": "AIME24", "amc12": "AMC12", "math500": "MATH500"}


def has_raw(raw: Path) -> bool:
    return all((raw / f"{b}.jsonl").is_file() for b in C.BENCHMARK_ORDER)


def raw_dirs(args) -> dict[str, dict[int, Path]]:
    """{method: {seed: raw dir}}: the hard arms, and uniform / ALG from E0 / E7 (seed 42) and W4 group C."""
    root, w4 = Path(args.root), Path(args.w4_root)
    out = {arm: {} for arm in ("uniform", "alg", *C.ARMS)}
    out["uniform"][42], out["alg"][42] = Path(args.uniform_run) / "raw", Path(args.alg_run) / "raw"
    for seed in C.NOISE_SEEDS:
        out["uniform"][seed] = w4 / "runs" / W4.run_id("uniform", "none", "none", seed) / "raw"
        out["alg"][seed] = w4 / "runs" / W4.run_id("alg", "P0", "S0", seed) / "raw"
    for arm in C.ARMS:
        for seed in (42, *C.NOISE_SEEDS):
            out[arm][seed] = root / "runs" / C.run_id(arm, seed) / "raw"
    return {method: {s: d for s, d in seeds.items() if has_raw(d)} for method, seeds in out.items()}


def metrics_of(rows: list[dict], grader: str) -> dict:
    column = GRADERS[grader]
    return evalreport.metrics([{**r, "correct_mathverify": r[column]} for r in rows])


def collect(args) -> dict:
    """{method: {seed: {"rows": graded samples, grader: metrics}}}"""
    out = {}
    for method, seeds in raw_dirs(args).items():
        out[method] = {}
        for seed, directory in seeds.items():
            rows = graded_rows(directory)
            out[method][seed] = {"rows": rows, "dir": directory, **{g: metrics_of(rows, g) for g in GRADERS}}
    return out


def avg(entry: dict, grader: str, key: str = "pass@1") -> float:
    return 100 * entry[grader]["average"][key]


def welch(a: list[float], b: list[float]) -> dict | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    if va + vb == 0:
        return None
    t = float((np.mean(a) - np.mean(b)) / math.sqrt(va + vb))
    df = float((va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)))
    try:
        from scipy.stats import t as student

        p = float(2 * student.sf(abs(t), df))
    except ImportError:
        p = None
    return {"t": t, "df": df, "p_two_sided": p}


def sigma(data: dict, grader: str, key: str) -> tuple[float | None, str]:
    values = {}
    for method in ("uniform", "alg"):
        seeds = data.get(method, {})
        if {42, *C.NOISE_SEEDS} <= set(seeds):
            values[method] = float(np.std([avg(seeds[s], grader, key) for s in (42, *C.NOISE_SEEDS)], ddof=1))
    if len(values) == 2:
        return max(values.values()), "w4 group C (seeds 42/43/44)"
    return None, "pending"


def paired_seed_mean(data, a: str, b: str, grader: str, seeds_a, seeds_b) -> dict:
    """Bootstrap of the difference of seed-averaged per-problem scores (every seed is graded; a problem's
    samples of all seeds are pooled, so pass@3 uses only the seed-42 runs)."""
    pool = lambda m, seeds: [r for s in seeds for r in data[m][s]["rows"]]  # noqa: E731
    return paired(pool(a, seeds_a), pool(b, seeds_b))[grader]


def decide(data: dict, grader: str, key: str) -> dict:
    """Section 7 on pass@1 (key) or pass@3."""
    short = "p1" if key == "pass@1" else "p3"
    sd, source = sigma(data, grader, key)
    threshold = max(1.0, 2 * sd) if sd is not None else 1.0
    out = {"metric": key, "grader": grader, "sigma": sd, "sigma_source": source, "threshold": threshold}
    if 42 not in data.get("h59", {}):
        return out | {"row": None, "pending": "H59 seed 42 has no evaluation yet"}
    alg_seeds, h_seeds = data.get("alg", {}), data["h59"]
    if 42 not in alg_seeds:
        published = W4.PUBLISHED["ALG default (P0, S0)"]["avg"][0 if key == "pass@1" else 1]
        return out | {"row": None, "pending": "no raw generations of the ALG seed-42 run: no paired comparison",
                      "alg_published": published, "h59": avg(h_seeds[42], grader, key)}
    three = {42, *C.NOISE_SEEDS}
    if three <= set(h_seeds) and three <= set(alg_seeds):
        a = [avg(alg_seeds[s], grader, key) for s in sorted(three)]
        h = [avg(h_seeds[s], grader, key) for s in sorted(three)]
        delta = float(np.mean(a) - np.mean(h))
        test = welch(a, h)
        significant = test is not None and test["p_two_sided"] is not None and test["p_two_sided"] < 0.05
        if delta >= threshold and significant:
            row = "soft_better"
        elif abs(delta) < 1.0:
            row = "equivalent"
        else:
            row = "leans_soft" if delta > 0 else "hard_better"
        return out | {"basis": "3-seed means, Welch", "alg": a, "h59": h, "delta": delta, "welch": test, "row": row,
                      "group_c_done": True}
    boot = paired(alg_seeds[42]["rows"], h_seeds[42]["rows"])[grader]
    delta = 100 * boot[f"delta_avg_{short}"]
    ci = (100 * boot[f"delta_avg_{short}_ci95_lo"], 100 * boot[f"delta_avg_{short}_ci95_hi"])
    return out | {"basis": "seed 42, paired bootstrap", "alg": avg(alg_seeds[42], grader, key),
                  "h59": avg(h_seeds[42], grader, key), "delta": delta, "ci95": ci,
                  "row": C.apply_rule(delta, threshold, ci), "group_c_done": False}


def read_decision(analysis: Path) -> dict:
    path = analysis / "decision.json"
    return C.read_json(path) if path.is_file() else {}


def cmd_rule(args) -> None:
    analysis = Path(args.root) / "analysis"
    if (analysis / "decision.json").is_file():
        print(f"{analysis / 'decision.json'} exists (the rule is written once)")
        return
    C.write_json(analysis / "decision.json",
                 {"rule": C.DECISION_RULE, "rule_written_at": C.now(), "status": "pending",
                  "headline_grader": args.grader, "provenance": C.provenance()})
    print(f"decision rule -> {analysis / 'decision.json'}")


def posthoc(root: Path) -> dict[str, dict[int, dict[int, dict]]]:
    """{method: {seed: {epoch: nll}}} from posthoc_nll/<method>__seed<s>/epoch_nll.json."""
    out = {}
    for path in sorted((root / "posthoc_nll").glob("*__seed*/epoch_nll.json")):
        method, seed = path.parent.name.split("__seed")
        epochs = C.read_json(path)["epochs"]
        out.setdefault(method, {})[int(seed)] = {int(e): v for e, v in epochs.items()}
    return out


def epoch_curve(root: Path, nll: dict, method: str) -> dict[int, float]:
    """{epoch: mean NLL on H59-dropped tokens}."""
    key = "dataset_nll_h59_dropped"
    values: dict[int, list[float]] = {}
    if method in ("uniform", "alg"):
        for epochs in nll.get(method, {}).values():
            for epoch, line in epochs.items():
                if key in line:
                    values.setdefault(epoch, []).append(line[key])
    else:
        for seed in (42, *C.NOISE_SEEDS):
            for line in read_jsonl(root / "runs" / C.run_id(method, seed) / "train/epoch_log.jsonl"):
                if key in line:
                    values.setdefault(line["epoch"], []).append(line[key])
    return {e: float(np.mean(v)) for e, v in sorted(values.items())}


def final_nll(root: Path, nll: dict, method: str) -> dict:
    keys = ("all", "prefix", "cont", "h59_dropped")
    lines = []
    if method in ("uniform", "alg"):
        lines = [epochs[max(epochs)] for epochs in nll.get(method, {}).values() if max(epochs) == 3]
    else:
        epochs = read_jsonl(root / "runs" / C.run_id(method, 42) / "train/epoch_log.jsonl")
        lines = epochs[-1:]
    return {k: float(np.mean([x[f"dataset_nll_{k}"] for x in lines if f"dataset_nll_{k}" in x]))
            if any(f"dataset_nll_{k}" in x for x in lines) else None for k in keys}


def markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def table_h1(data, stats, budget_source, grader) -> str:
    header = ["Method", "% supervised tokens", *[f"{BENCH_LABELS[b]} p@1/p@3" for b in C.BENCHMARK_ORDER],
              "Avg p@1", "Avg p@3", "Δ vs ALG, p@1 [CI 95%]", "Truncated", "Mean length", "seeds"]
    rows = []
    for method in ("uniform", "alg", *C.ARMS):
        name = {"uniform": "Uniform (P-ALIGN)", "alg": "ALG"}.get(method) or C.label(method, budget_source)
        supervised = "100%" if method in ("uniform", "alg") else (
            f"{stats[method]['kept_token_frac']:.1%}" if method in stats else "—")
        seeds = data.get(method, {})
        if not seeds:
            rows.append([name, supervised, *["—"] * 9, "0"])
            continue

        def cell(fn, seeds=seeds):
            values = [fn(e) for e in seeds.values()]
            if len(values) >= 3:
                return f"{np.mean(values):.2f}±{np.std(values, ddof=1):.2f}"
            return f"{np.mean(values):.2f}"

        bench = [cell(lambda e, b=b: 100 * e[grader][b]["pass@1"]) + " / "
                 + cell(lambda e, b=b: 100 * e[grader][b]["pass@3"]) for b in C.BENCHMARK_ORDER]
        delta = "—"
        if method != "alg" and 42 in seeds and 42 in data.get("alg", {}):
            boot = paired(seeds[42]["rows"], data["alg"][42]["rows"])[grader]
            delta = (f"{100 * boot['delta_avg_p1']:+.2f} [{100 * boot['delta_avg_p1_ci95_lo']:+.2f}, "
                     f"{100 * boot['delta_avg_p1_ci95_hi']:+.2f}]")
        truncated = np.mean([e[grader]["average"]["truncation_rate"] for e in seeds.values()])
        length = np.mean([e[grader]["average"]["mean_output_tokens"] for e in seeds.values()])
        rows.append([name, supervised, *bench, cell(lambda e: avg(e, grader)),
                     cell(lambda e: avg(e, grader, "pass@3")), delta, f"{truncated:.1%}", f"{length:.0f}",
                     ",".join(map(str, sorted(seeds)))])
    title = "Math-Verify OR OAT" if grader == "either" else "Math-Verify only"
    return f"### Table R-H1 ({title})\n\n" + markdown(header, rows)


def table_h2(root, stats, nll, budget_source) -> str:
    header = ["Method", "share: prefix", "share: continuation", "share: answer steps", "share: top-10% gain",
              "keeps final \\boxed step", "NLL all", "NLL prefix", "NLL continuation", "NLL H59-dropped"]
    reference = next(iter(stats.values()), {}).get("loss_coefficient_share_reference", {})
    rows = []
    for method in ("uniform", "alg", "h59", "r59", "h30", "alg-h59"):
        name = {"uniform": "Uniform", "alg": "ALG"}.get(method) or C.label(method, budget_source)
        share = reference.get(method) if method in ("uniform", "alg") else stats.get(method, {}).get(
            "loss_coefficient_share")
        shares = [f"{share[k]:.1%}" if share else "—"
                  for k in ("prefix", "continuation", "answer_steps", "top10pct_gain_steps")]
        boxed = "100.0%" if method in ("uniform", "alg") else (
            f"{stats[method]['pct_traces_keeping_final_boxed']:.1%}" if method in stats else "—")
        values = final_nll(root, nll, method)
        rows.append([name, *shares, boxed, *[f"{v:.4f}" if v is not None else "—" for v in values.values()]])
    return "### Table R-H2 (loss coefficients and final NLL on the 966 traces)\n\n" + markdown(header, rows)


def figures(root: Path, stats: dict, nll: dict, budget_source: str, overlap: dict | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sgl.positional.common import save_figure

    figure, axis = plt.subplots(figsize=(5, 3.4))
    for method in ("uniform", "alg", "h59", "r59"):
        curve = epoch_curve(root, nll, method)
        if curve:
            name = {"uniform": "Uniform", "alg": "ALG"}.get(method) or C.label(method, budget_source)
            axis.plot(list(curve), list(curve.values()), marker="o", label=name)
    axis.set_xlabel("epoch")
    axis.set_ylabel("NLL on tokens H59 drops")
    axis.set_xticks(range(4))
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    save_figure(figure, root / "analysis/fig_h1_dropped_nll")

    figure, axis = plt.subplots(figsize=(5, 3.4))
    positions = np.arange(1, 11)
    for arm in ("h59", "r59"):
        if arm in stats:
            axis.plot(positions, 100 * np.array(stats[arm]["kept_token_frac_by_position_decile"]), marker="o",
                      label=C.label(arm, budget_source))
    if overlap:
        axis.plot(positions, 100 * np.array(overlap["sgl_gate"]["kept_token_frac_by_position_decile"]), marker="s",
                  label="SGL spectral gate")
    axis.set_xlabel("relative position (decile)")
    axis.set_ylabel("% tokens kept")
    axis.set_xticks(positions)
    if axis.get_legend_handles_labels()[0]:
        axis.legend()
    save_figure(figure, root / "analysis/fig_h2_kept_by_position")


def master_rows(root: Path, data: dict, stats: dict, budget_source: str) -> list[dict]:
    rows = []
    for method, seeds in data.items():
        for seed, entry in sorted(seeds.items()):
            run = Path(entry["dir"]).parent
            summary_path = run / "train/train_summary.json"
            summary = C.read_json(summary_path) if summary_path.is_file() else {}
            manifest = C.read_json(run / "manifest.json") if (run / "manifest.json").is_file() else {}
            row = {"run_id": run.name, "arm": method, "label": C.label(method, budget_source), "seed": seed,
                   "probe": "P0", "seg": "S0",
                   "keep_frac": stats.get(method, {}).get("kept_token_frac", 1.0 if method in ("uniform", "alg")
                                                          else None)}
            for grader in GRADERS:
                suffix = "" if grader == "either" else "_mathverify"
                for name in (*C.BENCHMARK_ORDER, "average"):
                    row[f"{name}_p1{suffix}"] = 100 * entry[grader][name]["pass@1"]
                    row[f"{name}_p3{suffix}"] = 100 * entry[grader][name]["pass@3"]
            average = entry["either"]["average"]
            row |= {"truncation_rate": average["truncation_rate"], "mean_output_tokens": average["mean_output_tokens"],
                    "sample_agreement": average["sample_agreement"],
                    "train_hours": summary.get("wall_clock_hours"), "peak_mem_gb": summary.get("peak_mem_gb"),
                    "commit": (manifest.get("git") or {}).get("commit")}
            rows.append(row)
    return rows


def cmd_report(args) -> None:
    root = Path(args.root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    stats = {arm: C.read_json(root / "selection" / arm / "selection_stats.json") for arm in C.ARMS
             if (root / "selection" / arm / "selection_stats.json").is_file()}
    budget_source = next(iter(stats.values()), {}).get("budget_source", "fixed")
    data = collect(args)
    nll = posthoc(root)
    overlap_path = analysis / "selection_overlap.json"
    overlap = C.read_json(overlap_path) if overlap_path.is_file() else None

    # per-run paired bootstraps (each arm vs E0 / E7, plus the plan's pairs between arms)
    baselines = {"uniform": Path(args.uniform_run) / "raw", "alg": Path(args.alg_run) / "raw"}
    extra = {"h59": ("r59", "h30"), "alg-h59": ("h59",)}
    for run in sorted((root / "runs").glob("*__seed*")):
        meta = C.parse_run_id(run.name)
        others = {o: root / "runs" / C.run_id(o, meta["seed"]) / "raw" for o in extra.get(meta["arm"], ())}
        write_paired(run, baselines | others)

    tables = [table_h1(data, stats, budget_source, g) for g in sorted(GRADERS, key=lambda g: g != args.grader)]
    published = W4.PUBLISHED
    note = ("\nPublished (P-ALIGN's grader as in the paper), for reference: uniform "
            f"{published['Uniform (P-ALIGN)']['avg'][0]} / {published['Uniform (P-ALIGN)']['avg'][1]}, ALG "
            f"{published['ALG default (P0, S0)']['avg'][0]} / {published['ALG default (P0, S0)']['avg'][1]}.\n"
            f"SGL (s1K), uncontrolled comparison: {args.sgl_row or 'not filled in (pass --sgl-row)'}.\n")
    (analysis / "table_r_h1.md").write_text("\n".join(tables) + note)
    (analysis / "table_r_h2.md").write_text(table_h2(root, stats, nll, budget_source))
    figures(root, stats, nll, budget_source, overlap)
    pd.DataFrame(master_rows(root, data, stats, budget_source)).to_csv(analysis / "results_master.csv", index=False)

    decision = read_decision(analysis) or {"rule": C.DECISION_RULE, "rule_written_at": None}
    p1, p3 = decide(data, args.grader, "pass@1"), decide(data, args.grader, "pass@3")
    secondary = {}
    for name, a, b in (("h59_minus_r59", "h59", "r59"), ("h59_minus_uniform", "h59", "uniform"),
                       ("h59_minus_h30", "h59", "h30"), ("alg_h59_minus_alg", "alg-h59", "alg"),
                       ("alg_h59_minus_h59", "alg-h59", "h59")):
        if 42 in data.get(a, {}) and 42 in data.get(b, {}):
            boot = paired(data[a][42]["rows"], data[b][42]["rows"])[args.grader]
            secondary[name] = {k: 100 * v for k, v in boot.items() if k.startswith("delta")}
    secondary["nll_h59_dropped_by_epoch"] = {m: epoch_curve(root, nll, m) for m in ("uniform", "alg", "h59", "r59")}
    row = p1.get("row")
    if row is None:
        status, action = "pending", p1.get("pending")
    elif p1.get("group_c_done"):
        status, action = "final", C.DECISION_RULE["rows"].get(row)
    else:
        status = "needs group C" if C.needs_group_c(row) else "final"
        action = C.DECISION_RULE["rows"][row]
    decision |= {"status": status, "headline_grader": args.grader, "pass@1": p1, "pass@3": p3,
                 "pass@1_and_pass@3_agree": None if p3.get("row") is None else p1.get("row") == p3.get("row"),
                 "conclusion": row, "next_action": action, "needs_group_c": status == "needs group C",
                 "secondary": secondary, "decided_at": C.now(), "provenance": C.provenance()}
    C.write_json(analysis / "decision.json", decision)
    print((analysis / "table_r_h1.md").read_text())
    print(f"decision: {status}: {row} (delta {p1.get('delta')}, threshold {p1.get('threshold')}, "
          f"sigma {p1.get('sigma_source')}) -> {analysis / 'decision.json'}")
    missing = [m for m in ("uniform", "alg") if 42 not in data.get(m, {})]
    if missing:
        print(f"WARNING: no raw generations for the seed-42 {missing} run (--uniform-run / --alg-run)")


def cmd_needs_c(args) -> None:
    raise SystemExit(0 if read_decision(Path(args.root) / "analysis").get("needs_group_c") else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("rule", "report", "needs-c"))
    parser.add_argument("--root", default="results/hard")
    parser.add_argument("--w4-root", default="results/w4")
    parser.add_argument("--uniform-run", default="results/e0-uniform-r1-qwen-1.5b-palign")
    parser.add_argument("--alg-run", default="results/e7-alg-tau1-r1-qwen-1.5b-palign")
    parser.add_argument("--grader", choices=sorted(GRADERS), default="either")
    parser.add_argument("--sgl-row", help="text of the uncontrolled SGL (s1K) row")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {"rule": cmd_rule, "report": cmd_report, "needs-c": cmd_needs_c}[args.action](args)


if __name__ == "__main__":
    main()
