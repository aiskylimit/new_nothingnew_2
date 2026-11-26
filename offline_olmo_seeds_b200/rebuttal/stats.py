"""Statistics for the rebuttal analysis (numpy only, unit-tested in
tests/test_rebuttal_stats.py).

Two sources of noise matter here and both are resampled:
  * training seeds  - each seed is an independent training run;
  * problems        - each benchmark has only 30 problems.
The crossed bootstrap draws, per replicate, one resample of the problems of
every benchmark (shared by all methods, so method differences are PAIRED on
problems) and, independently for every method, one resample of its seeds.
A method's score in a replicate is the mean over the resampled seeds of the
benchmark-averaged accuracy on the resampled problems (each benchmark counts
equally, as in the paper's "Avg." column).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------


def argmax_earliest(scores: dict[int, float]) -> int:
    """Step with the highest score; ties go to the EARLIEST step (fewest tokens)."""
    if not scores:
        raise ValueError("no candidate checkpoints")
    best = max(scores.values())
    return min(s for s, v in scores.items() if np.isclose(v, best, rtol=0, atol=1e-12))


# ---------------------------------------------------------------------------
# Crossed (seed x problem) bootstrap
# ---------------------------------------------------------------------------


@dataclass
class RunScores:
    """Per-problem accuracies (n_correct / k) of one method under one selection
    protocol. benches[b] is an array [n_seeds, n_problems_b] whose rows are seeds
    and whose columns are the SAME problems, in the same order, for every method."""
    benches: dict[str, np.ndarray]

    @property
    def n_seeds(self) -> int:
        return next(iter(self.benches.values())).shape[0]

    def point(self) -> float:
        return float(np.mean([a.mean() for a in self.benches.values()]))

    def per_seed(self) -> np.ndarray:
        return np.mean([a.mean(axis=1) for a in self.benches.values()], axis=0)


def crossed_bootstrap(runs: dict[str, RunScores], n_boot: int = 10_000, seed: int = 0) -> dict[str, np.ndarray]:
    """Returns {method: array[n_boot]} of bootstrap replicates of the
    benchmark-averaged score. Problem draws are shared across methods."""
    rng = np.random.default_rng(seed)
    benches = sorted(next(iter(runs.values())).benches)
    for m, r in runs.items():
        if sorted(r.benches) != benches:
            raise ValueError(f"{m} has benchmarks {sorted(r.benches)}, expected {benches}")
    n_prob = {b: next(iter(runs.values())).benches[b].shape[1] for b in benches}
    prob_idx = {b: rng.integers(0, n_prob[b], size=(n_boot, n_prob[b])) for b in benches}
    out = {}
    for m, r in runs.items():
        s_idx = rng.integers(0, r.n_seeds, size=(n_boot, r.n_seeds))
        per_bench = []
        for b in benches:
            x = r.benches[b]                                   # [S, P]
            draws = x[s_idx[:, :, None], prob_idx[b][:, None, :]]  # [B, S, P]
            per_bench.append(draws.mean(axis=(1, 2)))
        out[m] = np.mean(per_bench, axis=0)
    return out


def problem_only_bootstrap(r: RunScores, n_boot: int = 10_000, seed: int = 0) -> np.ndarray:
    """Seeds held fixed, problems resampled: isolates benchmark-size noise."""
    rng = np.random.default_rng(seed)
    per_bench = []
    for b, x in sorted(r.benches.items()):
        idx = rng.integers(0, x.shape[1], size=(n_boot, x.shape[1]))
        per_bench.append(x[:, idx].mean(axis=(0, 2)))
    return np.mean(per_bench, axis=0)


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def diff_summary(a: np.ndarray, b: np.ndarray, point_a: float, point_b: float, level: float = 0.95) -> dict:
    """Bootstrap summary of a - b (replicates must come from the same
    crossed_bootstrap call so the problem draws are shared)."""
    d = a - b
    lo, hi = ci(d, level)
    p_le = float(np.mean(d <= 0))
    p_ge = float(np.mean(d >= 0))
    return dict(diff=point_a - point_b, ci_lo=lo, ci_hi=hi,
                p_two_sided=min(1.0, 2 * min(p_le, p_ge)), p_a_not_better=p_le)


def seed_bootstrap_ratio(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Ratio of means mean(x)/mean(y) with a bootstrap CI over seeds (x, y
    resampled independently), e.g. tokens-to-selected-checkpoint TROPIC/OPSD."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    ys = y[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(axis=1)
    lo, hi = ci(xs / ys)
    return float(x.mean() / y.mean()), lo, hi


# ---------------------------------------------------------------------------
# Seed-level inference (used when some seed only has aggregate scores, so the
# problems cannot be resampled): Student-t interval over seeds, Welch t-test.
# ---------------------------------------------------------------------------


def _t_pdf(x: float, df: float) -> float:
    from math import exp, lgamma, log, pi

    return exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * log(df * pi) - (df + 1) / 2 * log(1 + x * x / df))


def t_cdf(x: float, df: float) -> float:
    """Student-t CDF by Simpson integration of the density (no scipy needed)."""
    try:
        from scipy.stats import t as _t
        return float(_t.cdf(x, df))
    except ImportError:
        pass
    a = min(abs(x), 200.0)
    n = 4000
    h = a / n
    s = _t_pdf(0.0, df) + _t_pdf(a, df)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * _t_pdf(i * h, df)
    half = s * h / 3
    return 0.5 + half if x >= 0 else 0.5 - half


def t_ppf(q: float, df: float) -> float:
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(q, df))
    except ImportError:
        pass
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def t_interval(values, level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, float)
    if len(v) < 2:
        return float("nan"), float("nan")
    half = t_ppf(1 - (1 - level) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - half), float(v.mean() + half)


def welch_diff(a, b, level: float = 0.95) -> dict:
    """mean(a) - mean(b) over seeds with a Welch t interval and two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(a.mean() - b.mean())
    if len(a) < 2 or len(b) < 2:
        return dict(diff=d, ci_lo=float("nan"), ci_hi=float("nan"), p_two_sided=float("nan"),
                    p_a_not_better=float("nan"))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if se == 0:
        return dict(diff=d, ci_lo=d, ci_hi=d, p_two_sided=0.0 if d else 1.0, p_a_not_better=0.0 if d > 0 else 1.0)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tq = t_ppf(1 - (1 - level) / 2, df)
    tstat = d / se
    return dict(diff=d, ci_lo=d - tq * se, ci_hi=d + tq * se, p_two_sided=float(2 * (1 - t_cdf(abs(tstat), df))),
                p_a_not_better=float(1 - t_cdf(tstat, df)))


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), as in Spearman's rho."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(len(v), dtype=float)
    for val in np.unique(v):
        tie = v == val
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks
