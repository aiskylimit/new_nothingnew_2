"""Apply the decision rule fixed before the runs (decision.json) and write the result tables (report.md)."""

import argparse
import json
from pathlib import Path

from sgl.positional import common as C

RHO_LIMIT, NOISE = 0.3, 1.0  # median |rho|; Avg points below which a difference is noise


def apply_rule(median_abs_rho: float, p1: float, p3: float) -> tuple[str, list[str]]:
    """The table of section 3 of the plan; p1/p3 are R6's averages in percent."""
    uniform_p1 = C.PUBLISHED["P-ALIGN (uniform)"]["avg"][0]
    alerd_p1, alerd_p3 = C.PUBLISHED["ALERD"]["avg"]
    if abs(alerd_p1 - p1) < NOISE and abs(alerd_p3 - p3) < NOISE:
        return "R6 about equal to ALERD: soften the claim, run R7 and R8", ["r7", "r8"]
    if median_abs_rho >= RHO_LIMIT:
        return "substantial rho: run R7", ["r7"]
    if p1 <= uniform_p1 + NOISE:
        return "small rho, R6 about uniform or lower: enough, no further run", []
    if alerd_p1 - p1 >= NOISE and alerd_p3 - p3 >= NOISE:
        return "small rho, R6 above uniform and clearly below ALERD: enough; report that position helps partly", []
    return "NOT COVERED by the pre-registered table: decide by hand and say so", []


def avg(metrics: dict) -> tuple[float, float]:
    return 100 * metrics["average"]["pass@1"], 100 * metrics["average"]["pass@3"]


def load_metrics(root: Path, run: str):
    path = root / f"runs/{run}/eval/metrics.json"
    return C.read_json(path) if path.is_file() else None


def decide(root: Path) -> dict:
    rho = C.read_json(root / "analysis/a2_summary.json")["median_abs_rho"]
    distance = C.read_json(root / "analysis/a1b_summary.json")["L1_distance_R(alerd, r6)"]
    metrics = load_metrics(root, "r6")
    decision = {
        "median_abs_rho": rho, "a1b_L1_alerd_r6": distance, "r6_avg_p1": None, "r6_avg_p3": None,
        "r6_minus_uniform_p1": None, "alerd_minus_r6_p1": None, "alerd_minus_r6_p3": None,
        "rule_row_triggered": None, "next_action": None, "decided_at": None,
        "thresholds": {"median_abs_rho": RHO_LIMIT, "noise_points": NOISE, "published": C.PUBLISHED},
        "provenance": C.provenance(),
    }
    if metrics is None:
        decision["next_action"] = ["train and evaluate r6 first"]
        return decision
    p1, p3 = avg(metrics)
    row, runs = apply_rule(rho, p1, p3)
    decision.update({
        "r6_avg_p1": p1, "r6_avg_p3": p3,
        "r6_minus_uniform_p1": p1 - C.PUBLISHED["P-ALIGN (uniform)"]["avg"][0],
        "alerd_minus_r6_p1": C.PUBLISHED["ALERD"]["avg"][0] - p1,
        "alerd_minus_r6_p3": C.PUBLISHED["ALERD"]["avg"][1] - p3,
        "rule_row_triggered": row, "next_action": runs, "decided_at": C.now(),
    })
    return decision


def report(root: Path) -> str:
    lines = ["# Positional-bias check", "", "## Analysis", "", "| Quantity | Value |", "|---|---|"]
    a = {n: C.read_json(root / f"analysis/{n}_summary.json") for n in ("a1", "a2", "a3", "a4")}
    pct = lambda x: f"{100 * x:.1f}%" if x is not None else "-"  # noqa: E731
    lines += [
        f"| Traces used (K >= {C.MIN_STEPS}, non-constant gain) | {a['a2']['n_traces_used']} |",
        f"| A1: deciles with R > 1 / R by decile | {a['a1']['n_deciles_R_gt_1']} / "
        + " ".join(f"{x:.2f}" for x in a["a1"]["R_by_decile"]) + " |",
        f"| A2: median rho / median abs rho | {a['a2']['median_rho']:.3f} / {a['a2']['median_abs_rho']:.3f} |",
        f"| A2: outside chance range, positive / negative (expected 5% total) | "
        f"{pct(a['a2']['frac_outside_pos'])} / {pct(a['a2']['frac_outside_neg'])} |",
        f"| A3 (i): top-gain step in last 10% (random by token) | {pct(a['a3']['frac_top_in_last10'])} "
        f"({pct(a['a3']['expected_top_in_last10'])}) |",
        f"| A3 (ii): top-gain step is the last step | {pct(a['a3']['frac_top_is_last'])} |",
        f"| A3 (iii): late steps with w < 1, by step / token | {pct(a['a3']['late_frac_w_lt1_by_step'])} / "
        f"{pct(a['a3']['late_frac_w_lt1_by_token'])} |",
        f"| A4: prefix token mass, uniform / ALERD | {pct(a['a4'].get('prefix_token_share_unif'))} / "
        f"{pct(a['a4'].get('prefix_mass_share_alerd'))} |",
        "", "## Training (1.5B, 966 traces, math_verify, pass@1 / pass@3)", "",
        "| Method | AIME25 | AIME24 | AMC12 | MATH500 | Average |", "|---|---|---|---|---|---|",
    ]
    cell = lambda pair: f"{pair[0]:.2f} / {pair[1]:.2f}"  # noqa: E731
    order = (*C.BENCHMARK_ORDER, "avg")
    for name, values in C.PUBLISHED.items():
        lines.append(f"| {name} (published) | " + " | ".join(cell(values[b]) for b in order) + " |")
    for run, label in (("palign", "P-ALIGN (this report's generations)"), ("alerd", "ALERD (this report's generations)"),
                       ("r6", "**Position-only, increasing (R6)**"), ("r7", "Profile-matched (R7)"),
                       ("r8", "Stratified permutation (R8)")):
        metrics = load_metrics(root, run)
        if metrics is None:
            lines.append(f"| {label} | - | - | - | - | - |")
            continue
        cells = [cell((100 * metrics[b]["pass@1"], 100 * metrics[b]["pass@3"])) for b in C.BENCHMARK_ORDER]
        lines.append(f"| {label} | " + " | ".join([*cells, cell(avg(metrics))]) + " |")
    sanity = C.read_json(root / "analysis/sanity_weights.json")
    lines += ["", "## Weights (step percentiles)", "", "| Scheme | P10 | P50 | P90 | max |", "|---|---|---|---|---|"]
    for scheme in ("alerd", "r6", "r7", "r8"):
        s = sanity[scheme]
        lines.append(f"| {scheme} | {s['w_p10']:.3f} | {s['w_p50']:.3f} | {s['w_p90']:.3f} | {s['w_max']:.3f} |")
    path = root / "decision.json"
    if path.is_file():
        decision = C.read_json(path)
        lines += ["", "## Decision", "", f"{decision['rule_row_triggered']} (next: {decision['next_action']})"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/positional")
    parser.add_argument("--report-only", action="store_true", help="refresh report.md, keep decision.json")
    parser.add_argument("--print-next", action="store_true", help="print the runs decision.json asks for")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    path = root / "decision.json"
    if args.print_next:
        runs = C.read_json(path)["next_action"] if path.is_file() else []
        print(" ".join(r for r in runs or [] if r in ("r7", "r8")))
        return
    if not args.report_only:
        decision = decide(root)
        if path.is_file():  # an earlier decision is evidence: keep it
            old = C.read_json(path)
            if old.get("decided_at"):
                C.write_json(root / f"meta/decision-{old['decided_at'].replace(':', '')}.json", old)
        C.write_json(path, decision)
        print(json.dumps({k: decision[k] for k in ("median_abs_rho", "r6_avg_p1", "r6_avg_p3",
                                                   "rule_row_triggered", "next_action")}))
    (root / "report.md").write_text(report(root))
    print(f"report -> {root / 'report.md'}")


if __name__ == "__main__":
    main()
