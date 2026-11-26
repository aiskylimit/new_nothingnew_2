"""Figures for the multi-seed rebuttal results (reads the CSVs analyze.py writes).

    PYTHONPATH=. python rebuttal/plots.py --analysis analysis

Writes analysis/figures/<name>_<model>.{pdf,png}:
    main_forest        each method: mean +- std over seeds and every seed, one panel per protocol
    trajectory         avg@12 per benchmark vs step, mean +- std over seeds        (replaces Table 13)
    token_efficiency   Avg. vs generated rollout tokens, mean +- std over seeds      (replaces Figure 1a)
    fig1_seeds         Figure 1 (a) accuracy vs tokens and (b) rollout length vs step, mean +- std over seeds
    seeds              every seed's own curve, one panel per method
    dynamics           drift from the initial model, step size, rollout length, ...  (late-training analysis)
    coverage           pass@12 vs step, mean +- std                                 (replaces Figure 4)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal.presets import METHOD_LABELS, METHOD_ORDER  # noqa: E402

# Categorical slots validated for adjacent-pair CVD separation (dataviz reference
# palette, slots 1-4). Two of them are below 3:1 contrast on white, so every
# series also gets its own marker and a direct label / legend entry.
COLORS = {"tropic": "#2a78d6", "opsd": "#eb6834", "rlsd": "#1baf7a", "sdpo": "#eda100"}
MARKERS = {"tropic": "o", "opsd": "s", "rlsd": "^", "sdpo": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BENCH_LABELS = {"aime24": "AIME24", "aime25": "AIME25", "aime26": "AIME26", "hmmt25": "HMMT25", "avg": "Average"}
PROTOCOL_TITLES = {"paper": "Best ckpt per benchmark\n(Table 2 protocol)", "best_ckpt": "Best single checkpoint",
                   "final": "Last evaluated checkpoint"}


def base_of(m: str) -> str:
    return m.split("-")[0]


def color(m):
    return COLORS.get(base_of(m), INK2)


def marker(m):
    return MARKERS.get(base_of(m), "o")


def label(m: str) -> str:
    b, _, v = m.partition("-")
    return METHOD_LABELS.get(b, b) + (f" ({v})" if v else "")


def order(methods):
    return sorted(methods, key=lambda m: (METHOD_ORDER.index(base_of(m)) if base_of(m) in METHOD_ORDER else 99, m))


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "figure.dpi": 150, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })
    return plt


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None)
    print(f"  {out / name}.pdf")


def legend_below(fig, axes, methods, ncol=None):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=color(m), marker=marker(m), label=label(m)) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(methods), frameon=False,
               bbox_to_anchor=(0.5, -0.02))


def end_labels(ax, xs_ys: dict, dy_min: float):
    """Direct labels at the right end of each line, nudged apart so they never overlap."""
    items = sorted(((y, m, x) for m, (x, y) in xs_ys.items() if np.isfinite(y)), key=lambda t: t[0])
    placed = []
    for y, m, x in items:
        if placed and y - placed[-1] < dy_min:
            y = placed[-1] + dy_min
        placed.append(y)
        ax.annotate(label(m), (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                    color=INK, annotation_clip=False)


# ---------------------------------------------------------------------------


def fig_main_forest(plt, main, model, out, per_seed=None):
    """Mean over seeds (marker) +- 1 std (bar), with every seed as a small dot."""
    protos = [p for p in ("paper", "best_ckpt", "final") if p in set(main.protocol)]
    d = main[(main.model == model) & main.protocol.isin(protos)]
    if d.empty:
        return
    methods = order([m for m in d.method.unique() if m != "base"])
    base = main[(main.model == model) & (main.method == "base")]
    fig, axes = plt.subplots(1, len(protos), figsize=(2.3 * len(protos), 0.45 * len(methods) + 1.0), sharex=True,
                             squeeze=False)
    for ax, p in zip(axes[0], protos):
        dp = d[d.protocol == p]
        for i, m in enumerate(methods):
            r = dp[dp.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            y = len(methods) - 1 - i
            sd = 0.0 if pd.isna(r.avg_std) else r.avg_std
            ax.plot([(r.avg_mean - sd) * 100, (r.avg_mean + sd) * 100], [y, y], color=color(m), lw=2,
                    solid_capstyle="round")
            if per_seed is not None and not per_seed.empty:
                ps = per_seed[(per_seed.model == model) & (per_seed.protocol == p) & (per_seed.method == m)]
                ax.scatter(ps.avg * 100, np.full(len(ps), y - 0.22), s=9, color=color(m), alpha=0.7, lw=0)
            ax.plot(r.avg_mean * 100, y, marker=marker(m), color=color(m), ms=6, mec="white", mew=1.2)
            ax.annotate(f"{r.avg_mean * 100:.1f}", (r.avg_mean * 100, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
        if not base.empty:
            ax.axvline(base.iloc[0].avg_mean * 100, color=INK2, lw=0.8, ls=(0, (3, 2)))
            ax.annotate("base", (base.iloc[0].avg_mean * 100, len(methods) - 0.4), fontsize=6.5, color=INK2,
                        ha="center")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([label(m) for m in reversed(methods)])
        ax.set_ylim(-0.6, len(methods) - 0.2)
        ax.set_title(PROTOCOL_TITLES.get(p, p))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Average avg@12 (%)")
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle(f"{model}: mean over seeds, bar = +-1 std, dots = individual seeds", fontsize=8.5, y=1.12)
    save(fig, out, f"main_forest_{model}")
    plt.close(fig)


def fig_paired_diffs(plt, diffs, model, out):
    d = diffs[(diffs.model == model) & diffs.protocol.isin(["paper", "best_ckpt", "final"])]
    if d.empty:
        return
    protos = [p for p in ("paper", "best_ckpt", "final") if p in set(d.protocol)]
    baselines = order(d.b.unique())
    fig, axes = plt.subplots(1, len(protos), figsize=(2.4 * len(protos), 0.5 * len(baselines) + 1.1), sharex=True,
                             squeeze=False)
    for ax, p in zip(axes[0], protos):
        for i, bm in enumerate(baselines):
            r = d[(d.protocol == p) & (d.b == bm)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = len(baselines) - 1 - i
            ax.plot([r.ci_lo * 100, r.ci_hi * 100], [y, y], color=color(bm), lw=2, solid_capstyle="round")
            ax.plot(r["diff"] * 100, y, marker=marker(bm), color=color(bm), ms=6, mec="white", mew=1.2)
            ax.annotate(f"{r['diff'] * 100:+.1f}", (r["diff"] * 100, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(range(len(baselines)))
        ax.set_yticklabels([f"vs {label(b)}" for b in reversed(baselines)])
        ax.set_ylim(-0.6, len(baselines) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.set_title(PROTOCOL_TITLES.get(p, p))
        ax.set_xlabel("TROPIC - baseline (points)")
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle(f"{model}: paired difference in Average avg@12, bar = 95% CI", fontsize=8.5, y=1.12)
    save(fig, out, f"paired_diffs_{model}")
    plt.close(fig)


def fig_trajectory(plt, traj, model, benches, out):
    d = traj[(traj.model == model)]
    if d.empty:
        return
    methods = order(d.method.unique())
    panels = benches + ["avg"]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.75 * len(panels) + 0.6, 2.1), squeeze=False)
    for ax, b in zip(axes[0], panels):
        mcol, scol = ("avg_mean", "avg_std") if b == "avg" else (f"{b}_mean", f"{b}_std")
        ends = {}
        for m in methods:
            x = d[d.method == m].sort_values("step")
            y, s = x[mcol] * 100, x[scol].fillna(0) * 100
            ax.fill_between(x.step, y - s, y + s, color=color(m), alpha=0.15, lw=0)
            ax.plot(x.step, y, color=color(m), marker=marker(m))
            ends[m] = (x.step.iloc[-1], y.iloc[-1])
        ax.set_title(BENCH_LABELS.get(b, b))
        ax.set_xlabel("Training step")
    axes[0][0].set_ylabel("avg@12 (%)")
    legend_below(fig, axes, methods)
    fig.suptitle(f"{model}: mean over seeds, band = +-1 std over seeds", fontsize=8.5, y=1.03)
    fig.subplots_adjust(bottom=0.27, wspace=0.35)
    save(fig, out, f"trajectory_{model}")
    plt.close(fig)


def fig_token_efficiency(plt, traj, model, out, methods_keep=("tropic", "opsd")):
    d = traj[(traj.model == model) & traj.method.map(base_of).isin(methods_keep)]
    if d.empty or d.rollout_tokens_total.isna().all():
        return
    methods = order(d.method.unique())
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ends = {}
    for m in methods:
        x = d[d.method == m].sort_values("step")
        t = x.rollout_tokens_total / 1e6
        y, s = x.avg_mean * 100, x.avg_std.fillna(0) * 100
        ax.fill_between(t, y - s, y + s, color=color(m), alpha=0.15, lw=0)
        ax.plot(t, y, color=color(m), marker=marker(m))
        ends[m] = (t.iloc[-1], y.iloc[-1])
    end_labels(ax, ends, dy_min=0.6)
    ax.set_xlabel("Generated rollout tokens (millions)")
    ax.set_ylabel("Average avg@12 (%)")
    ax.set_title(f"{model}: mean over seeds, band = +-1 std", loc="left")
    save(fig, out, f"token_efficiency_{model}")
    plt.close(fig)


def fig_diff_by_step(plt, tdiff, model, out, baseline="opsd"):
    d = tdiff[(tdiff.model == model) & (tdiff.b == baseline)].sort_values("step")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.fill_between(d.step, d.ci_lo * 100, d.ci_hi * 100, color=color("tropic"), alpha=0.18, lw=0)
    ax.plot(d.step, d["diff"] * 100, color=color("tropic"), marker="o")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel("Training step (same step for both methods)")
    ax.set_ylabel(f"TROPIC - {label(baseline)} (points)")
    ax.set_title(f"{model}: paired difference, band = 95% CI", loc="left")
    save(fig, out, f"diff_by_step_{model}")
    plt.close(fig)


def fig_seeds(plt, ta, model, out):
    d = ta[(ta.model == model) & ta.complete & (ta.method != "base")]
    if d.empty:
        return
    methods = order(d.method.unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(1.7 * len(methods) + 0.5, 2.0), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        x = d[d.method == m]
        for seed, g in x.groupby("seed"):
            g = g.sort_values("step")
            ax.plot(g.step, g.avg_test * 100, color=color(m), lw=1, alpha=0.8, marker=marker(m), ms=2.5)
        mean = x.groupby("step").avg_test.mean() * 100
        ax.plot(mean.index, mean.values, color=INK, lw=1.6, ls=(0, (4, 2)))
        ax.set_title(f"{label(m)} ({x.seed.nunique()} seeds)")
        ax.set_xlabel("Training step")
    axes[0][0].set_ylabel("Average avg@12 (%)")
    fig.suptitle(f"{model}: one line per seed, dashed = mean", fontsize=8.5, y=1.04)
    save(fig, out, f"seeds_{model}")
    plt.close(fig)


FIG1_STYLE = {"tropic": dict(color="#1f5bbf", marker="s", ls="-"), "opsd": dict(color="#d62728", marker="o", ls="--")}
FIG1_STEPS = [10, 15, 20, 25, 40, 50, 75, 100]


def fig_paper_fig1(plt, traj, tr, model, out):
    """Figure 1 of the paper, averaged over training seeds: (a) average avg@12 vs generated rollout tokens,
    (b) mean tokens per rollout (cumulative: rollout_tokens_total / rollouts_total) vs step. Lines = mean over
    seeds, band = +-1 std. A point is drawn only when every seed of that method has it, so a method whose new
    seeds stop earlier than its seed 0 is not shown with fewer seeds at the end."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    drawn = []
    for m in ("tropic", "opsd"):
        st = FIG1_STYLE[m]
        d = traj[(traj.model == model) & (traj.method == m)] if not traj.empty else traj
        if not d.empty and "rollout_tokens_total" in d:
            full = d.n_seeds.max()
            x = d[((d.n_seeds == full) | (d.step == 0)) & d.rollout_tokens_total.notna()].sort_values("step")
            if len(x):
                t, y, s = x.rollout_tokens_total / 1e6, x.avg_mean * 100, x.avg_std.fillna(0) * 100
                ax1.fill_between(t, y - s, y + s, color=st["color"], alpha=0.15, lw=0)
                ax1.plot(t, y, color=st["color"], marker=st["marker"], ls=st["ls"], label=label(m))
                drawn.append(m)
        g = tr[(tr.model == model) & (tr.method == m)] if not tr.empty else tr
        if not g.empty and {"rollout_tokens_total", "rollouts_total"} <= set(g.columns):
            g = g[g.step.isin(FIG1_STEPS) & (g.rollouts_total > 0)].copy()
            g["len"] = g.rollout_tokens_total / g.rollouts_total
            # seeds whose logs have the token counters (older logs, e.g. the paper's OLMo seed 0, do not)
            n_all = g.seed.nunique()
            a = g.groupby("step").len.agg(["mean", "std", "count"])
            a = a[a["count"] == n_all]
            if len(a):
                ax2.fill_between(a.index, a["mean"] - a["std"].fillna(0), a["mean"] + a["std"].fillna(0),
                                 color=st["color"], alpha=0.15, lw=0)
                ax2.plot(a.index, a["mean"], color=st["color"], marker=st["marker"], ls=st["ls"],
                         label=None if m in drawn else label(m))
                drawn.append(m)
    if not drawn:
        plt.close(fig)
        return
    ax1.set_title("(a) Average avg@12")
    ax1.set_xlabel(r"Tokens generated ($\times 10^6$)")
    ax1.set_ylabel("Avg@12 accuracy (%)")
    ax2.set_title("(b) Rollout length")
    ax2.set_xlabel("Gradient update steps")
    ax2.set_ylabel("Mean tokens per rollout")
    for ax in (ax1, ax2):
        ax.grid(True, ls="--", alpha=0.5)
    h, lab = ax1.get_legend_handles_labels()
    h2, lab2 = ax2.get_legend_handles_labels()
    fig.legend(h + h2, lab + lab2, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(wspace=0.38, bottom=0.25)
    save(fig, out, f"fig1_seeds_{model}")
    plt.close(fig)


def fig_dynamics(plt, tr, model, out):
    d = tr[tr.model == model] if not tr.empty else tr
    if d.empty:
        return
    panels = [("kl_init_probe", "KL(current || initial)"), ("kl_step_probe", "KL(new || previous), per step"),
              ("mean_rollout_len", "Training rollout length"), ("frac_correct_rollouts", "Correct training rollouts"),
              ("mean_s", "TROPIC step size s"), ("G_k", "TROPIC privileged gap G")]
    panels = [(c, t) for c, t in panels if c in d.columns and d[c].notna().any()]
    if not panels:  # logs without any of these fields (e.g. a run still in progress)
        return
    methods = order(d.method.unique())
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 1.9 * nrow + 0.4), squeeze=False)
    for ax, (c, t) in zip(axes.flat, panels):
        for m in methods:
            x = d[(d.method == m) & d[c].notna()]
            if x.empty:
                continue
            g = x.groupby("step")[c]
            mean, sd = g.mean(), g.std().fillna(0)
            if len(mean) > 30:  # smooth per-step logs a little (window 5) for readability
                mean = mean.rolling(5, center=True, min_periods=1).mean()
                sd = sd.rolling(5, center=True, min_periods=1).mean()
            ax.fill_between(mean.index, mean - sd, mean + sd, color=color(m), alpha=0.15, lw=0)
            ax.plot(mean.index, mean.values, color=color(m), lw=1.4)
        ax.set_title(t)
        ax.set_xlabel("Training step")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    legend_below(fig, axes, methods)
    fig.suptitle(f"{model}: mean over seeds, band = +-1 std (rolling mean, window 5)", fontsize=8.5, y=1.01)
    fig.subplots_adjust(hspace=0.65, wspace=0.35, bottom=0.14)
    save(fig, out, f"dynamics_{model}")
    plt.close(fig)


def fig_coverage(plt, traj, model, out):
    d = traj[(traj.model == model) & traj.pass_mean.notna()]
    if d.empty:
        return
    methods = order(d.method.unique())
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ends = {}
    for m in methods:
        x = d[d.method == m].sort_values("step")
        y, s = x.pass_mean * 100, x.pass_std.fillna(0) * 100
        ax.fill_between(x.step, y - s, y + s, color=color(m), alpha=0.15, lw=0)
        ax.plot(x.step, y, color=color(m), marker=marker(m))
        ends[m] = (x.step.iloc[-1], y.iloc[-1])
    end_labels(ax, ends, dy_min=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Average pass@12 (%)")
    ax.set_title(f"{model}: mean over seeds, band = +-1 std", loc="left")
    save(fig, out, f"coverage_{model}")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", default=str(ROOT / "analysis"))
    ap.add_argument("--benches", nargs="+", default=None)
    args = ap.parse_args(argv)
    a = Path(args.analysis)
    plt = setup()

    def read(name):
        f = a / f"{name}.csv"
        return pd.read_csv(f) if f.exists() and f.stat().st_size > 1 else pd.DataFrame()

    main_df, per_seed, traj, ta, tr = (read(n) for n in
                                       ("main_results", "per_seed", "trajectory", "test_average", "train"))
    benches = args.benches or sorted({c[:-5] for c in traj.columns if c.endswith("_mean")} - {"avg", "pass"},
                                     key=lambda b: list(BENCH_LABELS).index(b) if b in BENCH_LABELS else 99)
    out = a / "figures"
    for model in sorted(set(traj.model) if not traj.empty else set(main_df.model)):
        print(f"[{model}]")
        fig_main_forest(plt, main_df, model, out, per_seed)
        fig_trajectory(plt, traj, model, benches, out)
        fig_token_efficiency(plt, traj, model, out)
        fig_paper_fig1(plt, traj, tr, model, out)
        fig_seeds(plt, ta, model, out)
        fig_dynamics(plt, tr, model, out)
        fig_coverage(plt, traj, model, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
