"""Rank stability of every W4 variant against the default (P0, S0) -- no training (plan section 2).

Probes share the S0 steps, so their u_k are compared directly, per trace:
  spearman_rho, top3_jaccard (top-3 steps by u), top3_pos_gain_share (the variant's own).
Segmentations have different steps, so w_k is spread over the step's tokens and compared per trace:
  token_spearman_rho, top10pct_mass_overlap (S0 weight mass of the tokens both put in their top 10%,
  over the S0 mass of S0's top 10%).
Random baseline, per trace: the same statistic for independent random orderings (probes: two random
rankings of the steps; segmentations: the variant's step weights shuffled across its steps).
`above_random95` = the trace's value exceeds the 95th percentile of its own random draws.

Writes analysis/rank_stability/<variant>.parquet (one row per trace), analysis/rank_stability_summary.csv
and analysis/table_r_w4a.md (median [IQR] plus the two W3 numbers from each signals/summary.json).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.robustness import common as C


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank().to_numpy()


def rho(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def top_set(values: np.ndarray, n: int = 3) -> set:
    return set(np.argsort(-values, kind="stable")[:n].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else math.nan


def token_weights(group: pd.DataFrame, length: int, column: str = "w") -> np.ndarray:
    out = np.empty(length)
    for first, last, weight in zip(group.tok_start, group.tok_end, group[column]):
        out[first:last] = weight
    return out


def top_mass_overlap(base: np.ndarray, other: np.ndarray, share: float = 0.1) -> float:
    a = base >= np.quantile(base, 1 - share)
    b = other >= np.quantile(other, 1 - share)
    return float(base[a & b].sum() / base[a].sum()) if base[a].sum() > 0 else math.nan


def probe_rows(base: pd.DataFrame, other: pd.DataFrame, n_random: int, rng) -> pd.DataFrame:
    rows = []
    others = dict(tuple(other.groupby("trace_id")))
    for trace_id, a in base.groupby("trace_id"):
        b = others.get(trace_id)
        if b is None or len(a) != len(b) or len(a) < 3:
            continue
        ua, ub = a.sort_values("k").u.to_numpy(), b.sort_values("k").u.to_numpy()
        K = len(ua)
        draws = [(rng.permutation(K).astype(float), rng.permutation(K).astype(float)) for _ in range(n_random)]
        rand_rho = np.array([rho(x, y) for x, y in draws])
        rand_jac = np.array([jaccard(top_set(x), top_set(y)) for x, y in draws])
        r, j = rho(ua, ub), jaccard(top_set(ua), top_set(ub))
        positive = np.clip(ub, 0, None)
        rows.append(
            {
                "trace_id": trace_id,
                "K": K,
                "spearman_rho": r,
                "top3_jaccard": j,
                "top3_pos_gain_share": float(positive[list(top_set(ub))].sum() / positive.sum())
                if positive.sum()
                else math.nan,
                "random_rho": float(rand_rho[0]),
                "random_jaccard": float(rand_jac[0]),
                "random_rho_q95": float(np.nanquantile(rand_rho, 0.95)),
                "random_jaccard_q95": float(np.nanquantile(rand_jac, 0.95)),
                "above_random95": bool(r > np.nanquantile(rand_rho, 0.95)) if not math.isnan(r) else False,
            }
        )
    return pd.DataFrame(rows)


def seg_rows(base: pd.DataFrame, other: pd.DataFrame, lengths: dict, n_random: int, rng) -> pd.DataFrame:
    rows = []
    others = dict(tuple(other.groupby("trace_id")))
    for trace_id, a in base.groupby("trace_id"):
        b = others.get(trace_id)
        if b is None:
            continue
        L = lengths[trace_id]
        wa, wb = token_weights(a, L), token_weights(b, L)
        r = rho(wa, wb)
        rand = []
        for _ in range(n_random):
            shuffled = b.assign(w=rng.permutation(b.w.to_numpy()))
            rand.append(rho(wa, token_weights(shuffled, L)))
        rand = np.array(rand)
        q95 = float(np.nanquantile(rand, 0.95)) if np.isfinite(rand).any() else math.nan
        rows.append(
            {
                "trace_id": trace_id,
                "K_base": len(a),
                "K_variant": len(b),
                "token_spearman_rho": r,
                "top10pct_mass_overlap": top_mass_overlap(wa, wb),
                "random_rho": float(rand[0]),
                "random_rho_q95": q95,
                "above_random95": bool(r > q95) if not (math.isnan(r) or math.isnan(q95)) else False,
            }
        )
    return pd.DataFrame(rows)


def iqr(series: pd.Series) -> tuple[float, float, float]:
    values = series.dropna()
    if values.empty:
        return math.nan, math.nan, math.nan
    return float(values.median()), float(values.quantile(0.25)), float(values.quantile(0.75))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/w4")
    parser.add_argument("--n-random", type=int, default=50, help="random draws per trace")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    signals = lambda p, s: root / "signals" / C.signal_name(p, s)  # noqa: E731
    base_dir = signals("P0", "S0")
    base = pd.read_parquet(base_dir / "steps.parquet")
    lengths = pd.read_parquet(base_dir / "traces.parquet").set_index("trace_id").num_response_tokens.to_dict()
    out = root / "analysis" / "rank_stability"
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for probe, seg in [("P0", "S0")] + [v for v in C.VARIANTS if v != ("P0", "S0")]:
        directory = signals(probe, seg)
        if not (directory / "steps.parquet").is_file():
            print(f"skip {probe}/{seg}: no {directory}/steps.parquet")
            continue
        stats = C.read_json(directory / "summary.json")
        row = {
            "variant": f"{probe}" if seg == "S0" else seg,
            "probe": probe,
            "seg": seg,
            "pos_gain_share_answer_steps": stats.get("pos_gain_share_answer_steps"),
            "pct_positive_steps_with_answer": stats.get("pct_positive_steps_with_answer"),
            "traces_top_step_has_answer": stats.get("traces_top_step_has_answer"),
            "n_steps": stats.get("n_steps"),
            "n_single_step_traces": stats.get("n_single_step_traces"),
        }
        if (probe, seg) != ("P0", "S0"):
            rng = np.random.default_rng(args.seed)
            other = pd.read_parquet(directory / "steps.parquet")
            if seg == "S0":
                table = probe_rows(base, other, args.n_random, rng)
                row["level"], rho_col, overlap_col = "step (u)", "spearman_rho", "top3_jaccard"
                row["random_overlap_median"] = float(table.random_jaccard.median())
            else:
                table = seg_rows(base, other, lengths, args.n_random, rng)
                row["level"], rho_col, overlap_col = "token (w)", "token_spearman_rho", "top10pct_mass_overlap"
            table["random_seed"] = args.seed
            table.to_parquet(out / f"{probe}__{seg}.parquet", index=False)
            row["rho_median"], row["rho_q25"], row["rho_q75"] = iqr(table[rho_col])
            row["overlap_median"], row["overlap_q25"], row["overlap_q75"] = iqr(table[overlap_col])
            row["overlap_metric"] = overlap_col
            row["random_rho_median"] = float(table.random_rho.median())
            row["pct_traces_above_random95"] = float(table.above_random95.mean())
            row["n_traces_compared"] = len(table)
        summary.append(row)
    frame = pd.DataFrame(summary)
    frame.to_csv(root / "analysis" / "rank_stability_summary.csv", index=False)

    fmt = lambda m, lo, hi: "–" if pd.isna(m) else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"  # noqa: E731
    pct = lambda x: "–" if x is None or pd.isna(x) else f"{100 * x:.1f}"  # noqa: E731
    lines = [
        "| Variant | Level | rho median [IQR] | Top-3 Jaccard / top-10% mass overlap | % traces > random 95% "
        "| % positive gain on answer steps | % traces whose top step has the answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in frame.to_dict("records"):
        lines.append(
            f"| {r['variant']} | {r.get('level', 'default')} "
            f"| {fmt(r.get('rho_median'), r.get('rho_q25'), r.get('rho_q75'))} "
            f"| {fmt(r.get('overlap_median'), r.get('overlap_q25'), r.get('overlap_q75'))} "
            f"| {pct(r.get('pct_traces_above_random95'))} | {pct(r['pos_gain_share_answer_steps'])} "
            f"| {pct(r['traces_top_step_has_answer'])} |"
        )
    probe_random = frame.dropna(subset=["random_rho_median"]) if "random_rho_median" in frame else frame.iloc[:0]
    if len(probe_random):
        lines.append(
            f"| Random (baseline) | – | {probe_random.random_rho_median.median():.2f} | "
            f"{frame.get('random_overlap_median', pd.Series(dtype=float)).median():.2f} | – | – | – |"
        )
    lines += [
        "",
        "Defaults quoted in the paper: 28.3% positive gain on answer steps, 66% of traces with the answer "
        "in the top step. Token-level columns for S1-S3 (plan 2.2).",
    ]
    (root / "analysis" / "table_r_w4a.md").write_text("\n".join(lines) + "\n")
    C.write_json(
        root / "analysis" / "rank_stability_meta.json",
        {"n_random": args.n_random, "seed": args.seed, "provenance": C.provenance()},
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
