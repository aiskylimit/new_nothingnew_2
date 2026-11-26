"""Multi-seed results with confidence intervals for the rebuttal.

    PYTHONPATH=. python rebuttal/analyze.py --runs-root runs --out analysis
    PYTHONPATH=. python rebuttal/plots.py   --analysis analysis

Every run (method x seed) has every checkpoint evaluated on every benchmark.
Each protocol picks checkpoints PER SEED, then results are aggregated over seeds:

    paper       best checkpoint per benchmark (the protocol of the paper's Table 2)   <- main
    best_ckpt   best single checkpoint by the 3-benchmark average (Table 13 / Figure 1)
    final       last evaluated checkpoint
    step<N>     a fixed step for every method (optional, --fixed-steps)

Uncertainty: `std` = standard deviation over training seeds; 95% CI = crossed
bootstrap over training seeds AND problems (rebuttal/stats.py). Method
differences are paired on problems. Note that "paper" and "best_ckpt" take a
maximum over checkpoints inside each seed; reporting the mean of these
per-seed maxima over 4 seeds (instead of one lucky run) is what makes them
comparable across methods.

Outputs in --out:
    per_problem.csv, checkpoints.csv, test_average.csv, train.csv   tidy data
    main_results.csv, main_results_<protocol>.tex                   Table 2 with mean +- std and CI
    per_seed.csv                                                    every seed's numbers (appendix)
    paired_diffs.csv                                                TROPIC - baseline, CI and p-value
    trajectory.csv, trajectory_diffs.csv                            per-step mean, std, CI (Figure 1 / 4)
    variance.csv                                                    seed noise vs problem noise
    cost.csv, compute.csv                                           tokens / teacher passes / GPU-h / FLOPs
    late_training.csv, dynamics.csv, genstats.csv                   decline after the peak, drift, lengths
    missing.csv                                                     evaluations still missing
    report.md                                                       everything above + draft sentences
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal.presets import METHOD_LABELS, METHOD_ORDER, TEST_BENCHMARKS  # noqa: E402
from rebuttal.stats import (  # noqa: E402
    RunScores, argmax_earliest, ci, crossed_bootstrap, diff_summary, problem_only_bootstrap, seed_bootstrap_ratio,
    spearman, t_interval, welch_diff,
)

# Parameter counts for the FLOPs estimate.
MODEL_PARAMS = {"qwen3-8b": 8.2e9, "olmo3-7b-think": 7.3e9, "qwen3-4b": 4.0e9, "smoke-qwen2.5-0.5b": 0.49e9}
BENCH_LABELS = {"aime24": "AIME24", "aime25": "AIME25", "aime26": "AIME26", "hmmt25": "HMMT25"}
MODEL_LABELS = {"qwen3-8b": "Qwen3-8B", "olmo3-7b-think": "Olmo-3-7B-Think", "qwen3-4b": "Qwen3-4B"}
PROTOCOL_LABELS = {
    "paper": "best checkpoint per benchmark, per seed (protocol of Table 2)",
    "best_ckpt": "best single checkpoint by the average, per seed (Table 13 / Figure 1)",
    "final": "last evaluated checkpoint",
}


def method_sort_key(m: str):
    base = m.split("-")[0]
    return (METHOD_ORDER.index(base) if base in METHOD_ORDER else 99, m)


def label(m: str) -> str:
    if m == "base":
        return "Base"
    base, _, var = m.partition("-")
    return METHOD_LABELS.get(base, base) + (f" ({var})" if var else "")


# ---------------------------------------------------------------------------
# 1. Collect
# ---------------------------------------------------------------------------


def collect(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (per-problem results, training logs, aggregate-only scores).

    Aggregate-only scores come from seed<k>/summary.json (written by import_scores.py) for runs whose
    generations are no longer available; they enter means/stds but not the problem-level bootstrap."""
    pp, tr, sm = [], [], []
    for model_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for method_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(method_dir.glob("seed*")):
                seed = int(seed_dir.name[4:])
                key = dict(model=model_dir.name, method=method_dir.name, seed=seed)
                for f in sorted(seed_dir.glob("eval/step*/*.json")):
                    d = json.loads(f.read_text())
                    step, bench = int(f.parent.name[4:]), f.stem
                    for i, pr in enumerate(d["problems"]):
                        s = pr["samples"]
                        toks = [x["n_tokens"] for x in s if x.get("n_tokens") is not None]
                        pp.append(dict(**key, step=step, bench=bench, idx=i, problem_id=pr["problem_id"], k=len(s),
                                       n_correct=sum(x["correct"] for x in s),
                                       mean_tokens=float(np.mean(toks)) if toks else np.nan,
                                       trunc_frac=_mean_known([x.get("truncated") for x in s]),
                                       noboxed_frac=float(np.mean([not x["has_boxed"] for x in s])),
                                       answer_entropy=pr.get("answer_entropy", np.nan)))
                log = seed_dir / "train_log.jsonl"
                if log.exists():
                    for line in log.read_text().splitlines():
                        if line.strip():
                            tr.append({**json.loads(line), **key})
                summ = seed_dir / "summary.json"
                if summ.exists():
                    d = json.loads(summ.read_text())
                    for step, by_bench in d["scores"].items():
                        for bench, v in by_bench.items():
                            sm.append(dict(**key, step=int(step), bench=bench, avg=float(v)))
    pp_df, tr_df, sm_df = pd.DataFrame(pp), pd.DataFrame(tr), pd.DataFrame(sm)
    if not pp_df.empty:
        pp_df["acc"] = pp_df["n_correct"] / pp_df["k"]
    return pp_df, tr_df, sm_df


def _mean_known(vals) -> float:
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan


def _col(g: pd.DataFrame, c: str) -> pd.Series:
    return g[c] if c in g.columns else pd.Series(np.nan, index=g.index)


def checkpoint_table(pp: pd.DataFrame, tr: pd.DataFrame, train_gpus: int, sm: pd.DataFrame | None = None) -> pd.DataFrame:
    keys = ["model", "method", "seed", "step", "bench"]
    if pp.empty:
        ck = pd.DataFrame(columns=keys + ["avg", "pass"])
    else:
        g = pp.groupby(keys)
        ck = g.agg(avg=("acc", "mean"), n_problems=("idx", "size"), mean_tokens=("mean_tokens", "mean"),
                   trunc_frac=("trunc_frac", "mean"), noboxed_frac=("noboxed_frac", "mean"),
                   answer_entropy=("answer_entropy", "mean")).reset_index()
        ck["pass"] = g["n_correct"].apply(lambda s: float((s > 0).mean())).values
    ck["summary_only"] = False
    if sm is not None and not sm.empty:
        have = set(map(tuple, ck[keys].itertuples(index=False))) if not ck.empty else set()
        extra = sm[[tuple(r) not in have for r in sm[keys].itertuples(index=False)]].copy()
        extra["summary_only"] = True
        ck = pd.concat([ck, extra], ignore_index=True)
    for c in ["n_problems", "mean_tokens", "trunc_frac", "noboxed_frac", "answer_entropy", "pass"]:
        if c not in ck.columns:
            ck[c] = np.nan
    ck["seed"] = ck["seed"].astype(int)
    ck["step"] = ck["step"].astype(int)
    cost_cols = ["rollout_tokens_total", "rollouts_total", "teacher_forwards_total", "teacher_tokens_total",
                 "student_tokens_total", "backward_tokens_total"]
    if not tr.empty:
        t = tr.sort_values("step").copy()
        t["wall_sec_cum"] = (t.groupby(["model", "method", "seed"])["elapsed_sec"].cumsum()
                             if "elapsed_sec" in t.columns else np.nan)
        keep = ["model", "method", "seed", "step", "wall_sec_cum"] + \
               [c for c in cost_cols + ["kl_init_probe"] if c in t.columns]
        ck = ck.merge(t[keep], on=["model", "method", "seed", "step"], how="left")
    for c in cost_cols + ["wall_sec_cum", "kl_init_probe"]:
        if c not in ck.columns:
            ck[c] = np.nan
        ck.loc[ck["step"] == 0, c] = 0.0
    ck["gpu_hours_cum"] = ck["wall_sec_cum"] / 3600 * train_gpus
    n = ck["model"].map(MODEL_PARAMS)
    # 2N FLOPs per token of every forward pass (generation, teacher, student scoring)
    # + 4N per back-propagated token; attention and prefill ignored (relative measure).
    ck["flops_cum"] = 2 * n * (ck["rollout_tokens_total"] + ck["teacher_tokens_total"] + ck["student_tokens_total"]) \
        + 4 * n * ck["backward_tokens_total"]
    return ck


def test_average(ck: pd.DataFrame, benches: list[str]) -> pd.DataFrame:
    sub = ck[ck["bench"].isin(benches)]
    idx = ["model", "method", "seed", "step"]
    wide = sub.pivot_table(index=idx, columns="bench", values="avg")
    have = [b for b in benches if b in wide.columns]
    wide["complete"] = wide[have].notna().all(axis=1) & (len(have) == len(benches))
    wide["avg_test"] = wide[have].mean(axis=1).where(wide["complete"])
    pw = sub.pivot_table(index=idx, columns="bench", values="pass")
    wide["pass_test"] = pw[[b for b in benches if b in pw.columns]].mean(axis=1).where(wide["complete"])
    return wide.reset_index()


# ---------------------------------------------------------------------------
# 2. Per-seed checkpoint choice under each protocol
# ---------------------------------------------------------------------------


def select(ck: pd.DataFrame, ta: pd.DataFrame, benches: list[str], fixed: list[int]) -> pd.DataFrame:
    """One row per (model, method, seed, protocol) with the chosen step for each benchmark."""
    rows = []
    for (model, method, seed), g in ck[ck.method != "base"].groupby(["model", "method", "seed"]):
        base = dict(model=model, method=method, seed=seed)
        steps = sorted(g.step.unique())
        per_bench = {}
        for b in benches:
            sc = g[g.bench == b].set_index("step")["avg"].to_dict()
            if sc:
                per_bench[b] = argmax_earliest(sc)
        if len(per_bench) == len(benches):
            rows.append(dict(**base, protocol="paper", **{f"step_{b}": s for b, s in per_bench.items()}))
        t = ta[(ta.model == model) & (ta.method == method) & (ta.seed == seed) & ta.complete]
        if not t.empty:
            s = argmax_earliest(t.set_index("step")["avg_test"].to_dict())
            rows.append(dict(**base, protocol="best_ckpt", **{f"step_{b}": s for b in benches}))
        rows.append(dict(**base, protocol="final", **{f"step_{b}": max(steps) for b in benches}))
        for fs in fixed:
            if fs in steps:
                rows.append(dict(**base, protocol=f"step{fs}", **{f"step_{b}": fs for b in benches}))
    sel = pd.DataFrame(rows)
    if not sel.empty:
        sel["last_step"] = sel[[f"step_{b}" for b in benches]].max(axis=1)
    return sel


def gather_scores(pp: pd.DataFrame, sel: pd.DataFrame, model: str, protocol: str, benches: list[str]):
    """{method: RunScores} (rows = seeds, columns = problems) and a list of missing evaluations."""
    out, missing = {}, []
    s = sel[(sel.model == model) & (sel.protocol == protocol)]
    sub = pp[pp.model == model]
    for method, g in s.groupby("method"):
        per_bench, seeds = {b: [] for b in benches}, []
        for _, r in g.sort_values("seed").iterrows():
            rows = {}
            for b in benches:
                step = int(r[f"step_{b}"])
                x = sub[(sub.method == method) & (sub.seed == r.seed) & (sub.step == step) & (sub.bench == b)]
                if x.empty:
                    missing.append(dict(model=model, method=method, seed=int(r.seed), step=step, bench=b))
                    rows = None
                    break
                rows[b] = x.sort_values("idx")["acc"].to_numpy()
            if rows is not None:
                seeds.append(int(r.seed))
                for b in benches:
                    per_bench[b].append(rows[b])
        if seeds:
            if any(len({len(a) for a in v}) != 1 for v in per_bench.values()):
                raise ValueError(f"{model}/{method}: seeds were evaluated on different numbers of problems")
            out[method] = (RunScores({b: np.vstack(v) for b, v in per_bench.items()}), seeds)
    if out:  # every method must have the same problems
        ref = {b: a.shape[1] for b, a in next(iter(out.values()))[0].benches.items()}
        out = {m: v for m, v in out.items() if {b: a.shape[1] for b, a in v[0].benches.items()} == ref}
    return out, missing


def scores_at_step(pp: pd.DataFrame, model: str, step: int, benches: list[str]) -> dict[str, RunScores]:
    """All seeds of every method at one fixed step (for the trajectory CIs)."""
    out = {}
    sub = pp[(pp.model == model) & (pp.step == step) & (pp.method != "base")]
    for method, g in sub.groupby("method"):
        per_bench = {}
        seeds = sorted(g.seed.unique())
        ok = True
        for b in benches:
            mats = []
            for sd in seeds:
                x = g[(g.seed == sd) & (g.bench == b)].sort_values("idx")["acc"].to_numpy()
                if len(x) == 0:
                    ok = False
                    break
                mats.append(x)
            if not ok or len({len(m) for m in mats}) != 1:
                ok = False
                break
            per_bench[b] = np.vstack(mats)
        if ok:
            out[method] = RunScores(per_bench)
    return out


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------


def fmt_pm(mean, std, scale=100, nd=1):
    if pd.isna(mean):
        return "--"
    return f"{mean * scale:.{nd}f}" if pd.isna(std) else f"{mean * scale:.{nd}f} $\\pm$ {std * scale:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "analysis"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--benches", nargs="+", default=TEST_BENCHMARKS)
    ap.add_argument("--fixed-steps", type=int, nargs="*", default=[])
    ap.add_argument("--reference", default="tropic", help="Method compared against every other method.")
    ap.add_argument("--baseline", default="opsd", help="Main baseline (cost ratio, draft sentences).")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--train-gpus", type=int, default=4, help="GPUs reserved per training run (1 trainer + vLLM).")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pp, tr, sm = collect(Path(args.runs_root))
    if pp.empty and sm.empty:
        raise SystemExit(f"no eval results under {args.runs_root}")
    if args.models:
        pp = pp[pp.model.isin(args.models)] if not pp.empty else pp
        tr = tr[tr.model.isin(args.models)] if not tr.empty else tr
        sm = sm[sm.model.isin(args.models)] if not sm.empty else sm
    tb = args.benches
    pp = pp[pp.bench.isin(tb)] if not pp.empty else pp
    sm = sm[sm.bench.isin(tb)] if not sm.empty else sm
    ck = checkpoint_table(pp, tr, args.train_gpus, sm)
    ta = test_average(ck, tb)
    sel = select(ck, ta, tb, args.fixed_steps)
    for name, df in [("per_problem", pp), ("checkpoints", ck), ("test_average", ta), ("train", tr), ("selections", sel)]:
        df.to_csv(out / f"{name}.csv", index=False)

    protocols = ["paper", "best_ckpt", "final"] + [f"step{s}" for s in args.fixed_steps]
    main_rows, seed_rows, diff_rows, var_rows, cost_rows, missing = [], [], [], [], [], []
    ck_cost = ck.drop_duplicates(["model", "method", "seed", "step"])[
        ["model", "method", "seed", "step", "rollout_tokens_total", "teacher_forwards_total", "gpu_hours_cum",
         "flops_cum"]]
    cell = ck.set_index(["model", "method", "seed", "step", "bench"])[["avg", "summary_only"]]
    BOOT = "bootstrap over seeds and problems"
    TINT = "t-interval over seeds (some cells are aggregate-only)"

    models = sorted(set(ck.model))
    for model in models:
        b = ck[(ck.model == model) & (ck.method == "base")]
        if b.bench.nunique() == len(tb):
            row = dict(model=model, protocol="base", method="base", n_seeds=1)
            for bench in tb:
                row[f"{bench}_mean"] = float(b[b.bench == bench].avg.mean())
            row["avg_mean"] = float(np.mean([row[f"{x}_mean"] for x in tb]))
            main_rows.append(row)
        for protocol in protocols:
            s = sel[(sel.model == model) & (sel.protocol == protocol)]
            if s.empty:
                continue
            # seed-level scores (work for per-problem and aggregate-only cells alike)
            seedtab, uses_summary = {}, False
            for method, g in s.groupby("method"):
                seeds, rows = [], []
                for _, r in g.sort_values("seed").iterrows():
                    vals = []
                    for bench in tb:
                        key = (model, method, int(r.seed), int(r[f"step_{bench}"]), bench)
                        if key not in cell.index:
                            missing.append(dict(model=model, method=method, seed=int(r.seed),
                                                step=int(r[f"step_{bench}"]), bench=bench))
                            vals = None
                            break
                        c = cell.loc[key]
                        uses_summary |= bool(c.summary_only)
                        vals.append(float(c.avg))
                    if vals is not None:
                        seeds.append(int(r.seed))
                        rows.append(vals)
                if seeds:
                    seedtab[method] = (seeds, np.array(rows))
            if not seedtab:
                continue
            runs, boots = {}, {}
            if not uses_summary:
                runs, miss = gather_scores(pp, sel, model, protocol, tb)
                missing += miss
                boots = crossed_bootstrap({m: v[0] for m, v in runs.items()}, n_boot=args.n_boot, seed=0) if runs else {}
            for m in sorted(seedtab, key=method_sort_key):
                seeds, mat = seedtab[m]
                per_seed = mat.mean(axis=1)
                n = len(seeds)
                full = (not uses_summary) and m in runs and runs[m][1] == seeds
                row = dict(model=model, protocol=protocol, method=m, n_seeds=n, avg_mean=float(per_seed.mean()),
                           avg_std=float(per_seed.std(ddof=1)) if n > 1 else np.nan, ci_method=BOOT if full else TINT)
                row["avg_ci_lo"], row["avg_ci_hi"] = ci(boots[m]) if full else t_interval(per_seed)
                for j, bench in enumerate(tb):
                    row[f"{bench}_mean"] = float(mat[:, j].mean())
                    row[f"{bench}_std"] = float(mat[:, j].std(ddof=1)) if n > 1 else np.nan
                    if full:
                        bb = crossed_bootstrap({m: RunScores({bench: runs[m][0].benches[bench]})},
                                               n_boot=args.n_boot, seed=2)[m]
                        row[f"{bench}_ci_lo"], row[f"{bench}_ci_hi"] = ci(bb)
                    else:
                        row[f"{bench}_ci_lo"], row[f"{bench}_ci_hi"] = t_interval(mat[:, j])
                main_rows.append(row)
                if full:
                    pb = problem_only_bootstrap(runs[m][0], n_boot=args.n_boot, seed=1)
                    var_rows.append(dict(model=model, protocol=protocol, method=m, n_seeds=n,
                                         seed_sd=row["avg_std"], problem_se=float(pb.std(ddof=1)),
                                         total_se=float(boots[m].std(ddof=1))))
                for i, sd in enumerate(seeds):
                    sr = s[(s.method == m) & (s.seed == sd)].iloc[0]
                    srow = dict(model=model, protocol=protocol, method=m, seed=sd, avg=float(per_seed[i]),
                                last_step=int(sr.last_step),
                                aggregate_only=any(bool(cell.loc[(model, m, sd, int(sr[f"step_{x}"]), x)].summary_only)
                                                   for x in tb))
                    for j, bench in enumerate(tb):
                        srow[bench] = float(mat[i, j])
                        srow[f"step_{bench}"] = int(sr[f"step_{bench}"])
                    seed_rows.append(srow)
            if args.reference in seedtab:
                ref_seeds = seedtab[args.reference][1].mean(axis=1)
                for m in sorted(seedtab, key=method_sort_key):
                    if m == args.reference:
                        continue
                    if not uses_summary and args.reference in boots and m in boots:
                        d = diff_summary(boots[args.reference], boots[m], float(ref_seeds.mean()),
                                         float(seedtab[m][1].mean(axis=1).mean()))
                        how = BOOT + " (paired on problems)"
                    else:
                        d = welch_diff(ref_seeds, seedtab[m][1].mean(axis=1))
                        how = "Welch t-test over seeds"
                    diff_rows.append(dict(model=model, protocol=protocol, a=args.reference, b=m, ci_method=how, **d))
            # cost to the last checkpoint each seed needs under this protocol
            c = s.merge(ck_cost, left_on=["model", "method", "seed", "last_step"],
                        right_on=["model", "method", "seed", "step"], how="left")
            for m, g in c.groupby("method"):
                g = g.dropna(subset=["rollout_tokens_total"])
                if g.empty:
                    continue
                cost_rows.append(dict(model=model, protocol=protocol, method=m, n_seeds=len(g),
                                      last_step_mean=g.last_step.mean(), last_step_max=g.last_step.max(),
                                      tokens_M_mean=g.rollout_tokens_total.mean() / 1e6,
                                      tokens_M_std=g.rollout_tokens_total.std(ddof=1) / 1e6 if len(g) > 1 else np.nan,
                                      teacher_fwd_mean=g.teacher_forwards_total.mean(),
                                      gpu_hours_mean=g.gpu_hours_cum.mean(), pflops_mean=g.flops_cum.mean() / 1e15))
            a = c[c.method == args.reference].rollout_tokens_total.dropna().to_numpy()
            bsl = c[c.method == args.baseline].rollout_tokens_total.dropna().to_numpy()
            if len(a) and len(bsl):
                r_, lo_, hi_ = seed_bootstrap_ratio(a, bsl, n_boot=args.n_boot)
                cost_rows.append(dict(model=model, protocol=protocol, method=f"ratio {args.reference}/{args.baseline}",
                                      n_seeds=min(len(a), len(bsl)), tokens_ratio=r_, tokens_ratio_lo=lo_,
                                      tokens_ratio_hi=hi_))

    main_df, seed_df, diff_df = pd.DataFrame(main_rows), pd.DataFrame(seed_rows), pd.DataFrame(diff_rows)
    var_df, cost_df = pd.DataFrame(var_rows), pd.DataFrame(cost_rows)
    miss_df = pd.DataFrame(missing).drop_duplicates() if missing else pd.DataFrame()
    for name, df in [("main_results", main_df), ("per_seed", seed_df), ("paired_diffs", diff_df),
                     ("variance", var_df), ("cost", cost_df), ("missing", miss_df)]:
        df.to_csv(out / f"{name}.csv", index=False)

    # ---- trajectories: mean, std and CI at every checkpoint step ----
    traj_rows, tdiff_rows = [], []
    tt = ta[ta.complete & (ta.method != "base")]
    for model in models:
        for step in sorted(tt[tt.model == model].step.unique()):
            at = tt[(tt.model == model) & (tt.step == step)]
            agg_only = bool(ck[(ck.model == model) & (ck.step == step) & (ck.method != "base")].summary_only.any())
            runs = {} if agg_only else scores_at_step(pp, model, step, tb)
            boots = crossed_bootstrap(runs, n_boot=args.n_boot, seed=3) if runs else {}
            per_method = {}
            for m, g in at.groupby("method"):
                ps = g.sort_values("seed").avg_test.to_numpy()
                per_method[m] = ps
                full = m in boots and runs[m].n_seeds == len(ps)
                lo, hi = ci(boots[m]) if full else t_interval(ps)
                row = dict(model=model, method=m, step=step, n_seeds=len(ps), avg_mean=float(ps.mean()),
                           avg_std=float(ps.std(ddof=1)) if len(ps) > 1 else np.nan, avg_ci_lo=lo, avg_ci_hi=hi,
                           ci_method=BOOT if full else TINT)
                for bench in tb:
                    x = g[bench].to_numpy()
                    row[f"{bench}_mean"] = float(x.mean())
                    row[f"{bench}_std"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
                traj_rows.append(row)
            if args.reference in per_method:
                for m in per_method:
                    if m == args.reference:
                        continue
                    if args.reference in boots and m in boots:
                        d = diff_summary(boots[args.reference], boots[m], float(per_method[args.reference].mean()),
                                         float(per_method[m].mean()))
                    else:
                        d = welch_diff(per_method[args.reference], per_method[m])
                    tdiff_rows.append(dict(model=model, step=step, a=args.reference, b=m, **d))
    traj_df, tdiff_df = pd.DataFrame(traj_rows), pd.DataFrame(tdiff_rows)
    if not traj_df.empty:
        cost_at = ck_cost.groupby(["model", "method", "step"]).rollout_tokens_total.mean().reset_index()
        traj_df = traj_df.merge(cost_at, on=["model", "method", "step"], how="left")
        p = ta[ta.complete & (ta.method != "base")].groupby(["model", "method", "step"]).pass_test.agg(
            pass_mean="mean", pass_std="std").reset_index()
        traj_df = traj_df.merge(p, on=["model", "method", "step"], how="left")
        base = ta[(ta.method == "base") & ta.complete]
        for _, r in base.iterrows():  # base model as step 0 of every method's curve
            for m in traj_df[traj_df.model == r.model].method.unique():
                traj_df = pd.concat([traj_df, pd.DataFrame([dict(model=r.model, method=m, step=0, n_seeds=1,
                                                                 avg_mean=r.avg_test, rollout_tokens_total=0.0,
                                                                 pass_mean=r.pass_test,
                                                                 **{f"{b}_mean": r[b] for b in tb})])],
                                    ignore_index=True)
        traj_df = traj_df.sort_values(["model", "method", "step"])
    traj_df.to_csv(out / "trajectory.csv", index=False)
    tdiff_df.to_csv(out / "trajectory_diffs.csv", index=False)

    # ---- compute per training step ----
    comp_rows = []
    if not tr.empty:
        for (model, method), g in tr.groupby(["model", "method"]):
            n = MODEL_PARAMS.get(model, np.nan)
            fl = 2 * n * (_col(g, "rollout_tokens") + _col(g, "teacher_tokens") + _col(g, "student_tokens")) \
                + 4 * n * _col(g, "backward_tokens")
            sec = _col(g, "elapsed_sec")
            comp_rows.append(dict(model=model, method=method, n_seeds=g.seed.nunique(),
                                  sec_per_step=sec.mean(),
                                  gpu_hours_per_100_steps=sec.mean() * 100 / 3600 * args.train_gpus,
                                  rollout_tokens_per_step=_col(g, "rollout_tokens").mean(),
                                  teacher_forwards_per_step=_col(g, "teacher_forwards").mean(),
                                  teacher_tokens_per_step=_col(g, "teacher_tokens").mean(),
                                  tflops_per_step=fl.mean() / 1e12))
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out / "compute.csv", index=False)

    # ---- late training: decline after the peak and its link to drift from the initial model ----
    late_rows = []
    tt = ta[ta.complete & (ta.method != "base")]
    kl = ck.drop_duplicates(["model", "method", "seed", "step"])[["model", "method", "seed", "step", "kl_init_probe"]]
    for (model, method), g in tt.groupby(["model", "method"]):
        curve = g.groupby("step").avg_test.mean()
        drops = [gs.sort_values("step").avg_test.max() - gs.sort_values("step").avg_test.iloc[-1]
                 for _, gs in g.groupby("seed")]
        j = g.merge(kl, on=["model", "method", "seed", "step"], how="left")
        late_rows.append(dict(model=model, method=method, n_seeds=g.seed.nunique(),
                              peak_step_of_mean=int(curve.idxmax()), peak_of_mean=float(curve.max()),
                              final_of_mean=float(curve.iloc[-1]), drop_mean=float(np.mean(drops)),
                              drop_std=float(np.std(drops, ddof=1)) if len(drops) > 1 else np.nan,
                              spearman_klinit_vs_acc=spearman(j.kl_init_probe, j.avg_test)))
    late_df = pd.DataFrame(late_rows)
    late_df.to_csv(out / "late_training.csv", index=False)

    dyn_cols = ["kl_init_probe", "kl_step_probe", "kl_step_last", "mean_rollout_len", "truncated_frac",
                "frac_correct_rollouts", "mean_s", "frac_binding", "G_k", "grad_norm"]
    dyn_rows = []
    if not tr.empty:
        for (model, method), g in tr.groupby(["model", "method"]):
            for step in (10, 25, 50, 75, 100):
                x = g[g.step == step]
                if x.empty:
                    continue
                row = dict(model=model, method=method, step=step, n_seeds=x.seed.nunique())
                for c in dyn_cols:
                    if c in x.columns and x[c].notna().any():
                        row[f"{c}_mean"] = x[c].mean()
                        row[f"{c}_std"] = x[c].std(ddof=1) if x[c].notna().sum() > 1 else np.nan
                dyn_rows.append(row)
    dyn_df = pd.DataFrame(dyn_rows)
    dyn_df.to_csv(out / "dynamics.csv", index=False)

    gen = ck.groupby(["model", "method", "step", "bench"]).agg(
        n_seeds=("seed", "nunique"), avg=("avg", "mean"), pass_at_k=("pass", "mean"), mean_tokens=("mean_tokens", "mean"),
        trunc_frac=("trunc_frac", "mean"), noboxed_frac=("noboxed_frac", "mean"),
        answer_entropy=("answer_entropy", "mean")).reset_index()
    gen.to_csv(out / "genstats.csv", index=False)

    write_latex(main_df, tb, out)
    write_report(out, args, main_df, diff_df, tdiff_df, var_df, cost_df, comp_df, late_df, dyn_df, miss_df)
    print(f"analysis written to {out}/ (start with report.md)")
    return 0


# ---------------------------------------------------------------------------
# 4. LaTeX + report
# ---------------------------------------------------------------------------


def write_latex(main_df: pd.DataFrame, tb: list[str], out: Path) -> None:
    """Paper Table 2 layout: mean +- std over training seeds per benchmark and for the average."""
    for protocol in [p for p in main_df.protocol.unique() if p != "base"]:
        d = main_df[main_df.protocol.isin([protocol, "base"])]
        lines = [r"\begin{tabular}{l" + "c" * (len(tb) + 1) + "}", r"\toprule",
                 "Method & " + " & ".join(BENCH_LABELS.get(b, b) for b in tb) + r" & Avg. \\", r"\midrule"]
        for model in d.model.unique():
            lines.append(rf"\multicolumn{{{len(tb) + 2}}}{{l}}{{\textit{{{MODEL_LABELS.get(model, model)}}}}} \\")
            dm = d[d.model == model]
            for m in sorted(dm.method.unique(), key=lambda x: (-1, x) if x == "base" else method_sort_key(x)):
                r = dm[dm.method == m].iloc[0]
                name = "Base" if m == "base" else "+ " + label(m)
                cells = [fmt_pm(r[f"{b}_mean"], r.get(f"{b}_std", np.nan)) for b in tb]
                cells.append(fmt_pm(r["avg_mean"], r.get("avg_std", np.nan)))
                lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / f"main_results_{protocol}.tex").write_text("\n".join(lines) + "\n")
        if protocol == "paper":
            write_appendix_table(lines, d, out)


def write_appendix_table(tabular: list[str], d: pd.DataFrame, out: Path) -> None:
    """Ready-to-\\input appendix table (Table 2 protocol, mean +- std over training seeds)."""
    seeds = sorted({int(x) for x in d[d.method != "base"].n_seeds.dropna()})
    n_txt = str(seeds[0]) if len(seeds) == 1 else f"{min(seeds)}--{max(seeds)}"
    caption = (r"\textbf{Robustness across training seeds.} Same protocol as Table~\ref{tab:main} (avg@12, best "
               r"checkpoint per benchmark), applied separately to each of " + n_txt + r" independent training runs "
               r"per method (different data order, LoRA initialization and rollout sampling). Entries are mean "
               r"$\pm$ standard deviation over training seeds.")
    body = [r"\begin{table}[t]", r"\centering", r"\small", r"\resizebox{\linewidth}{!}{%"] + tabular[:-1] + \
        [r"\end{tabular}}", r"\caption{" + caption + "}", r"\label{tab:seeds}", r"\end{table}"]
    (out / "appendix_seeds_table.tex").write_text("\n".join(body) + "\n")


def write_diff_latex(diff_df: pd.DataFrame, out: Path, reference: str) -> None:
    """Appendix table: reference - baseline on the average, per protocol, with 95% CI and p-value."""
    if diff_df.empty:
        return
    protos = [p for p in ("paper", "best_ckpt", "final") if p in set(diff_df.protocol)]
    short = {"paper": "Best per benchmark", "best_ckpt": "Best single ckpt", "final": "Last ckpt"}
    lines = [r"\begin{tabular}{ll" + "c" * len(protos) + "}", r"\toprule",
             "Model & Baseline & " + " & ".join(short[p] for p in protos) + r" \\", r"\midrule"]
    for model in diff_df.model.unique():
        d = diff_df[diff_df.model == model]
        for i, b in enumerate(sorted(d.b.unique(), key=method_sort_key)):
            cells = []
            for p in protos:
                r = d[(d.protocol == p) & (d.b == b)]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                cells.append(f"{r['diff'] * 100:+.1f} [{r.ci_lo * 100:+.1f}, {r.ci_hi * 100:+.1f}]")
            name = MODEL_LABELS.get(model, model) if i == 0 else ""
            lines.append(f"{name} & {label(b)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (out / "appendix_paired_diffs.tex").write_text(
        f"% {label(reference)} minus baseline, average avg@12 in points, 95% CI from the seeds x problems bootstrap\n"
        + "\n".join(lines) + "\n")


PCT = {"avg_mean", "avg_std", "avg_ci_lo", "avg_ci_hi", "diff", "ci_lo", "ci_hi", "seed_sd", "problem_se",
       "total_se", "peak_of_mean", "final_of_mean", "drop_mean", "drop_std"}


def md_table(df: pd.DataFrame, cols: list[str], pct=frozenset(), nd: int = 2) -> str:
    if df is None or df.empty:
        return "_(no data yet)_\n"
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if pd.isna(v) else (f"{v * 100:.1f}" if c in pct else f"{v:.{nd}f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(out, args, main_df, diff_df, tdiff_df, var_df, cost_df, comp_df, late_df, dyn_df, miss_df):
    tb = args.benches
    pct = PCT | {f"{b}_{s}" for b in tb for s in ("mean", "std", "ci_lo", "ci_hi")}
    L = ["# Multi-seed results (auto-generated)\n",
         f"Benchmarks: **{', '.join(BENCH_LABELS.get(b, b) for b in tb)}**; scores are avg@k in %. "
         f"Entries are mean and standard deviation (`std`) over training seeds. Checkpoints without generations "
         f"use their aggregate score (`summary.json`, e.g. taken from the paper); per_seed.csv marks them "
         f"`aggregate_only`.\n"]
    if not miss_df.empty:
        L.append(f"> **{len(miss_df)} evaluations are missing** (missing.csv); seeds with a missing evaluation are "
                 "left out of the affected protocol.\n")
    for protocol in [p for p in main_df.protocol.unique() if p != "base"]:
        d = main_df[main_df.protocol.isin([protocol, "base"])].copy()
        d["method"] = d["method"].map(label)
        L.append(f"\n## Results - {PROTOCOL_LABELS.get(protocol, protocol)}\n")
        L.append(md_table(d, ["model", "method", "n_seeds"] + sum([[f"{b}_mean", f"{b}_std"] for b in tb], [])
                          + ["avg_mean", "avg_std"], pct))
        ps = pd.read_csv(out / "per_seed.csv") if (out / "per_seed.csv").stat().st_size > 1 else pd.DataFrame()
        if not ps.empty:
            ps = ps[ps.protocol == protocol].copy()
            ps["method"] = ps["method"].map(label)
            L.append("\nPer seed:\n\n")
            L.append(md_table(ps, ["model", "method", "seed", "avg"] + tb + [f"step_{b}" for b in tb]
                              + ["aggregate_only"], {"avg"} | set(tb)))
    L.append("\n## Training cost up to the last checkpoint each protocol needs (mean over seeds)\n")
    L.append(md_table(cost_df[~cost_df.method.astype(str).str.startswith("ratio")] if not cost_df.empty else cost_df,
                      ["model", "protocol", "method", "n_seeds", "last_step_mean", "last_step_max",
                       "tokens_M_mean", "tokens_M_std", "teacher_fwd_mean", "gpu_hours_mean", "pflops_mean"]))
    L.append("\n## Compute per training step\n")
    L.append(md_table(comp_df, ["model", "method", "n_seeds", "sec_per_step", "gpu_hours_per_100_steps",
                                "rollout_tokens_per_step", "teacher_forwards_per_step", "teacher_tokens_per_step",
                                "tflops_per_step"], nd=1))
    L.append("\nFLOPs: 2N per token of every forward pass + 4N per back-propagated token; attention and prefill "
             "ignored, so compare methods with it rather than quoting absolute values.\n")
    L.append("\n## Late training (decline after the peak)\n")
    L.append(md_table(late_df, ["model", "method", "n_seeds", "peak_step_of_mean", "peak_of_mean", "final_of_mean",
                                "drop_mean", "drop_std", "spearman_klinit_vs_acc"], pct, 3))
    L.append("\n## Training dynamics (mean over seeds)\n")
    L.append(md_table(dyn_df, ["model", "method", "step", "n_seeds", "kl_init_probe_mean", "kl_step_probe_mean",
                               "mean_rollout_len_mean", "frac_correct_rollouts_mean", "mean_s_mean",
                               "frac_binding_mean", "G_k_mean"], nd=4))
    L.append(draft_sentences(main_df, diff_df, cost_df, args))
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def fmt_p(p: float, n_boot: int) -> str:
    floor = 2.0 / n_boot  # smallest two-sided p-value the bootstrap can resolve
    return f"p < {floor:.0e}" if p < floor else f"p = {p:.3f}"


def draft_sentences(main_df, diff_df, cost_df, args) -> str:
    S = ["\n## Draft sentences for the response (check every number before use)\n"]
    for model in main_df.model.unique():
        for protocol in ("paper", "best_ckpt", "final"):
            v = main_df[(main_df.model == model) & (main_df.protocol == protocol)]
            a, b = v[v.method == args.reference], v[v.method == args.baseline]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            txt = (f"- **{model}, {PROTOCOL_LABELS[protocol]}**: over {int(a.n_seeds)} training seeds, "
                   f"{label(args.reference)} {a.avg_mean * 100:.1f} ± {a.avg_std * 100:.1f} vs "
                   f"{label(args.baseline)} {b.avg_mean * 100:.1f} ± {b.avg_std * 100:.1f} (mean ± std).")
            c = cost_df[(cost_df.model == model) & (cost_df.protocol == protocol)
                        & cost_df.method.astype(str).str.startswith("ratio")] if not cost_df.empty else cost_df
            if not c.empty:
                txt += (f" Its checkpoints need {c.iloc[0].tokens_ratio:.2f}x the generated rollout tokens of "
                        f"{label(args.baseline)}'s (mean over seeds).")
            S.append(txt)
    return "\n".join(S) + "\n"


if __name__ == "__main__":
    sys.exit(main())
