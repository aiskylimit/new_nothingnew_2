"""A1-A4 on the step and trace tables (CPU, no training). Every figure comes with its data as CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.positional import common as C
from sgl.positional.tables import profile_R

N_BOOT, N_PERM = 1000, 1000
DECILES = list(range(C.N_DECILES))


def _per_trace_deciles(index: np.ndarray, decile: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    table = np.zeros((n, C.N_DECILES))
    np.add.at(table, (index, decile), values)
    return table


def decile_mass(trace_ids, decile, ell, weight, n_boot: int = N_BOOT, seed: int = 42) -> pd.DataFrame:
    """mass_unif, mass_scheme and R(d), with a bootstrap-over-traces interval of R."""
    _, index = np.unique(trace_ids, return_inverse=True)
    n = index.max() + 1
    decile, ell, weight = np.asarray(decile), np.asarray(ell, float), np.asarray(weight, float)
    uniform = _per_trace_deciles(index, decile, ell, n)
    scheme = _per_trace_deciles(index, decile, ell * weight, n)

    def ratio(counts):
        u, s = counts @ uniform, counts @ scheme
        u = u / u.sum(-1, keepdims=True)
        s = s / s.sum(-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return u, s, s / u

    mass_unif, mass_scheme, R = ratio(np.ones(n))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)
    boot = ratio(counts)[2]
    return pd.DataFrame({
        "decile": DECILES,
        "n_steps": np.bincount(decile, minlength=C.N_DECILES),
        "n_tokens": uniform.sum(0).astype(int),
        "mass_unif": mass_unif, "mass_scheme": mass_scheme, "R": R,
        "R_ci_lo": np.nanpercentile(boot, 2.5, axis=0), "R_ci_hi": np.nanpercentile(boot, 97.5, axis=0),
    })


def spearman(x, y) -> float:
    a, b = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def a1(steps, out: Path, seed: int):
    import matplotlib.pyplot as plt

    alerd = decile_mass(steps.trace_id, steps.decile, steps.ell, steps.w_alerd, seed=seed)
    uniform = decile_mass(steps.trace_id, steps.decile, steps.ell, np.ones(len(steps)), seed=seed)
    pd.concat([uniform.assign(scheme="uniform"), alerd.assign(scheme="alerd")])[
        ["scheme", "decile", "n_steps", "n_tokens", "mass_unif", "mass_scheme", "R", "R_ci_lo", "R_ci_hi"]
    ].to_csv(out / "a1_decile_mass.csv", index=False)
    C.write_json(out / "a1_summary.json", {
        "R_by_decile": alerd.R.tolist(),
        "n_deciles_R_gt_1": int((alerd.R > 1).sum()),
        "spearman_R_vs_decile": spearman(DECILES, alerd.R),
        "mass_last_decile_unif": float(alerd.mass_unif.iloc[-1]),
        "mass_last_decile_alerd": float(alerd.mass_scheme.iloc[-1]),
        "bootstrap_n": N_BOOT, "bootstrap_seed": seed,
    })
    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(C.N_DECILES)
    axis.bar(x - 0.2, alerd.mass_unif, 0.4, label="uniform", color="#bbbbbb")
    axis.bar(x + 0.2, alerd.mass_scheme, 0.4, label="ALERD", color="#1f77b4")
    axis.set(xlabel="position decile", ylabel="share of weight mass", xticks=x)
    right = axis.twinx()
    right.errorbar(x, alerd.R, yerr=[alerd.R - alerd.R_ci_lo, alerd.R_ci_hi - alerd.R], color="#d62728",
                   marker="o", capsize=2, label="R(d)")
    right.axhline(1.0, color="#d62728", linestyle=":", linewidth=1)
    right.set_ylabel("R(d) = ALERD / uniform")
    axis.legend(loc="upper left")
    C.save_figure(figure, out / "a1_decile_mass")


def a1b(steps, out: Path, seed: int):
    import matplotlib.pyplot as plt

    frames = {s: decile_mass(steps.trace_id, steps.decile, steps.ell, steps[f"w_{s}"], seed=seed)
              for s in ("alerd", "r6", "r7")}
    pd.concat([f.assign(scheme=s) for s, f in frames.items()]).to_csv(out / "a1b_profiles.csv", index=False)
    difference = np.abs(frames["alerd"].R - frames["r6"].R)
    C.write_json(out / "a1b_summary.json", {
        "L1_distance_R(alerd, r6)": float(difference.sum()),
        "max_abs_diff_decile": int(difference.idxmax()),
        "max_abs_diff": float(difference.max()),
        "L1_distance_R(alerd, r7)": float(np.abs(frames["alerd"].R - frames["r7"].R).sum()),
        # the plan fixes no threshold for "very different": a human reads the figure
        "needs_r7_by_profile": None,
        "needs_r7_note": "judge a1b_profiles.png before training results are seen; rerun with RUNS_EXTRA=r7 if so",
    })
    figure, axis = plt.subplots(figsize=(7, 4))
    for scheme, label in (("alerd", "ALERD"), ("r6", "R6 position-only"), ("r7", "R7 profile-matched")):
        axis.plot(DECILES, frames[scheme].R, marker="o", label=label)
    axis.axhline(1.0, color="#888888", linestyle=":")
    axis.set(xlabel="position decile", ylabel="R(d)", xticks=DECILES)
    axis.legend()
    C.save_figure(figure, out / "a1b_profiles")


def a2(steps, out: Path, seed: int):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    rows, null_all, constant, short = [], [], 0, 0
    for trace_id, trace in steps.groupby("trace_id"):
        K = len(trace)
        if K < C.MIN_STEPS:
            short += 1
            continue
        rho = spearman(trace.u, trace.p)
        if np.isnan(rho):  # constant gain: no ranking to correlate
            constant += 1
            continue
        base = np.tile(np.arange(K, dtype=float) - (K - 1) / 2, (N_PERM, 1))
        left, right = rng.permuted(base, axis=1), rng.permuted(base, axis=1)
        null = (left * right).sum(1) / (base[0] ** 2).sum()
        low, high = np.percentile(null, [2.5, 97.5])
        null_all.append(null)
        rows.append({"trace_id": trace_id, "K": K, "rho": rho, "rho_index": spearman(trace.u, trace.k),
                     "null_lo": low, "null_hi": high, "outside_pos": rho > high, "outside_neg": rho < low})
    table = pd.DataFrame(rows)
    table.to_csv(out / "a2_rho_per_trace.csv", index=False)
    pooled = np.concatenate(null_all)
    pd.DataFrame({"rho_null": pooled}).to_csv(out / "a2_null_pooled.csv", index=False, float_format="%.5f")
    C.write_json(out / "a2_summary.json", {
        "n_traces_used": len(table), "n_excluded": short + constant,
        "n_excluded_short": short, "n_excluded_constant_gain": constant,
        "median_rho": float(table.rho.median()),
        "iqr_rho": [float(table.rho.quantile(0.25)), float(table.rho.quantile(0.75))],
        "median_abs_rho": float(table.rho.abs().median()),
        "frac_rho_pos": float((table.rho > 0).mean()),
        "frac_outside_pos": float(table.outside_pos.mean()),
        "frac_outside_neg": float(table.outside_neg.mean()),
        "frac_outside_total": float((table.outside_pos | table.outside_neg).mean()),
        "expected_outside": 0.05,
        "median_rho_index": float(table.rho_index.median()),
        "rho_index_note": "p_k increases strictly with k, so the rank correlation with k equals the one with p",
        "null_n_perm": N_PERM, "null_seed": seed,
    })
    figure, axis = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1, 1, 41)
    axis.hist(pooled, bins=bins, density=True, color="#bbbbbb", label="chance (permutations)")
    axis.hist(table.rho, bins=bins, density=True, color="#1f77b4", alpha=0.7, label="answer gain vs position")
    axis.axvline(table.rho.median(), color="#d62728", label=f"median = {table.rho.median():.2f}")
    axis.set(xlabel="Spearman rho within a trace", ylabel="density")
    axis.legend()
    C.save_figure(figure, out / "a2_rho_hist")


def a3(steps, traces, out: Path):
    import matplotlib.pyplot as plt

    included = traces[~traces.excluded_A2A3 & ~traces.sigma_u_zero]
    kept = steps[steps.trace_id.isin(included.trace_id)]
    share_late10 = kept.assign(t=kept.ell * (kept.p > 0.9)).groupby("trace_id").t.sum() / \
        kept.groupby("trace_id").ell.sum()
    late, early = kept[kept.p > 0.8], kept[kept.p <= 0.8]

    def below_one(frame):
        return {"by_step": float((frame.w_alerd < 1).mean()),
                "by_token": float(frame.ell[frame.w_alerd < 1].sum() / frame.ell.sum())}

    C.write_json(out / "a3_summary.json", {
        "n_traces_used": len(included),
        "frac_top_in_last10": float((included.argmax_p > 0.9).mean()),
        "expected_top_in_last10": float(share_late10.mean()),
        "frac_top_is_last": float(included.argmax_is_last.mean()),
        "expected_top_is_last": float((1 / included.K).mean()),
        "late_steps_n": len(late),
        "late_frac_w_lt1_by_step": below_one(late)["by_step"],
        "late_frac_w_lt1_by_token": below_one(late)["by_token"],
        "early_steps_n": len(early),
        "early_frac_w_lt1_by_step": below_one(early)["by_step"],
        "early_frac_w_lt1_by_token": below_one(early)["by_token"],
        "late_threshold_p": 0.8,
    })
    bins = np.linspace(0, 1, 11)
    observed = np.histogram(included.argmax_p, bins=bins)[0] / len(included)
    reference = kept.ell.groupby([kept.trace_id, kept.decile]).sum() / kept.groupby(["trace_id"]).ell.sum()
    reference = reference.unstack(fill_value=0).reindex(columns=DECILES, fill_value=0).mean()
    pd.DataFrame({"decile": DECILES, "frac_top_step": observed, "random_by_token": reference.values}).to_csv(
        out / "a3_argmax_pos_hist.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(DECILES, observed, color="#1f77b4", label="top-gain step")
    axis.plot(DECILES, reference.values, color="#d62728", marker="o", label="random by token mass")
    axis.set(xlabel="position decile", ylabel="share of traces", xticks=DECILES)
    axis.legend()
    C.save_figure(figure, out / "a3_argmax_pos_hist")
    with (out / "a3_late_examples.jsonl").open("w") as handle:
        for row in late.nsmallest(20, "w_alerd").itertuples():
            handle.write(json.dumps({"trace_id": int(row.trace_id), "k": int(row.k), "p": float(row.p),
                                     "u": float(row.u), "w_alerd": float(row.w_alerd), "text": row.text_head},
                                    ensure_ascii=False) + "\n")


def segment_pieces(steps, traces) -> pd.DataFrame:
    """Split every step at the marker; positions are relative within the prefix / continuation."""
    marked = traces[traces.n_tok_prefix >= 0].set_index("trace_id")
    frame = steps[steps.trace_id.isin(marked.index)].copy()
    n_prefix = frame.trace_id.map(marked.n_tok_prefix)
    total = frame.trace_id.map(marked.L)
    pieces = []
    for segment, column, offset, length in (
        ("prefix", "n_tok_prefix", 0 * n_prefix, n_prefix),
        ("continuation", "n_tok_cont", n_prefix, total - n_prefix),
    ):
        part = frame[frame[column] > 0].copy()
        part["n"] = part[column]
        first = np.maximum(part.tok_start, offset[part.index])
        position = (first - offset[part.index] + part.n / 2) / length[part.index]
        part["segment_decile"] = [C.decile(p) for p in position]
        part["segment"] = segment
        pieces.append(part[["trace_id", "segment", "segment_decile", "n", "w_alerd"]])
    return pd.concat(pieces)


def a4(steps, traces, out: Path, seed: int):
    import matplotlib.pyplot as plt

    pieces = segment_pieces(steps, traces)
    if pieces.empty:
        C.write_json(out / "a4_summary.json", {"n_traces_with_marker": 0, "note": "no <End_of_Prefix> marker"})
        return
    pieces["mass"] = pieces.n * pieces.w_alerd
    prefix = pieces[pieces.segment == "prefix"]
    per_trace = pd.DataFrame({
        "unif": prefix.groupby("trace_id").n.sum() / pieces.groupby("trace_id").n.sum(),
        "alerd": prefix.groupby("trace_id").mass.sum() / pieces.groupby("trace_id").mass.sum(),
    }).dropna()
    mean_w = pieces.groupby("segment").mass.sum() / pieces.groupby("segment").n.sum()
    C.write_json(out / "a4_summary.json", {
        "n_traces_with_marker": int(pieces.trace_id.nunique()),
        "prefix_token_share_unif": float(prefix.n.sum() / pieces.n.sum()),
        "prefix_mass_share_alerd": float(prefix.mass.sum() / pieces.mass.sum()),
        "median_trace_prefix_token_share_unif": float(per_trace.unif.median()),
        "median_trace_prefix_mass_share_alerd": float(per_trace.alerd.median()),
        "n_steps_with_marker": int(steps.has_marker.sum()),
        "mean_w_prefix_tokens": float(mean_w.get("prefix", np.nan)),
        "mean_w_cont_tokens": float(mean_w.get("continuation", np.nan)),
    })
    frames = []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (segment, part) in zip(axes, pieces.groupby("segment", sort=False)):
        table = decile_mass(part.trace_id, part.segment_decile, part.n, part.w_alerd, seed=seed)
        frames.append(table.rename(columns={"mass_scheme": "mass_alerd", "decile": "decile"}).assign(segment=segment))
        axis.errorbar(DECILES, table.R, yerr=[table.R - table.R_ci_lo, table.R_ci_hi - table.R], marker="o", capsize=2)
        axis.axhline(1.0, color="#888888", linestyle=":")
        axis.set(title=segment, xlabel="decile within segment", xticks=DECILES)
    axes[0].set_ylabel("R(d)")
    pd.concat(frames)[["segment", "decile", "mass_unif", "mass_alerd", "R", "R_ci_lo", "R_ci_hi"]].to_csv(
        out / "a4_decile_mass_by_segment.csv", index=False)
    C.save_figure(figure, out / "a4_by_segment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/positional")
    parser.add_argument("--seed", type=int, default=42, help="bootstrap and permutation null")
    return parser


def main(argv: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    step_path, trace_path = root / "tables/step_table.parquet", root / "tables/trace_table.parquet"
    steps, traces = pd.read_parquet(step_path), pd.read_parquet(trace_path)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    a1(steps, out, args.seed)
    a1b(steps, out, args.seed)
    a2(steps, out, args.seed)
    a3(steps, traces, out)
    a4(steps, traces, out, args.seed)
    C.write_json(root / "meta/analysis.json", {
        **C.provenance({"step_table": step_path, "trace_table": trace_path}), "seed": args.seed,
        "R_alerd": profile_R(steps, "w_alerd").tolist(),
    })
    for name in ("a1", "a1b", "a2", "a3", "a4"):
        summary = C.read_json(out / f"{name}_summary.json")
        print(f"--- {name}: " + json.dumps({k: v for k, v in summary.items() if not k.endswith("note")}, default=float))


if __name__ == "__main__":
    main()
