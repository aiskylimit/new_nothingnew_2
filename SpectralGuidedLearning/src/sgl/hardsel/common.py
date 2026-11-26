"""Arms, token tags, run names and the pre-registered decision rule of the ALG-hard experiment.

Layout under results/hard/:
  selection/<arm>/    steps.parquet, traces.parquet, select_check.json, selection_stats.json, manifest.json
  runs/<run_id>/      config.yaml, train/, checkpoints/, raw/, eval/, timing.json, manifest.json, run_check.json
  posthoc_nll/<tag>/  epoch_nll.json (uniform / ALG checkpoints)
  analysis/           tables, figures, results_master.csv, decision.json, selection_overlap.json
"""

from sgl.positional.common import BENCHMARK_ORDER, now, provenance, read_json, sha256, write_json  # noqa: F401
from sgl.robustness.common import PUBLISHED, STUDENT, git_state, hardware, versions  # noqa: F401

ARMS = ("h59", "r59", "h30", "alg-h59")
GROUP_A, GROUP_B = ("h59", "r59"), ("h30", "alg-h59")
FIXED_FRACTION = {"h59": 0.59, "h30": 0.30}  # r59 and alg-h59 follow h59 trace by trace
RANDOM_SEED = 42  # R59's per-trace step shuffle: Random(RANDOM_SEED + trace id)
NOISE_SEEDS = (43, 44)

# token_tags: bits 1..32 are those of sgl.robustness.signals (prefix, continuation, junction, answer step,
# stop token, any step token); 64 / 128 = this arm keeps / drops the token; 256 = H59 drops it (the same
# token set for every arm, so their NLL on it can be compared).
TAG_KEPT, TAG_DROPPED, TAG_H59_DROPPED = 64, 128, 256


def label(arm: str, budget_source: str = "fixed") -> str:
    """Name used in tables; with per-trace SGL budgets the H59 family is the plan's H-SGL."""
    sgl = budget_source == "sgl_gate"
    return {
        "h59": "H-SGL" if sgl else "H59",
        "r59": "R-SGL" if sgl else "R59",
        "h30": "H30",
        "alg-h59": "ALG∘H-SGL" if sgl else "ALG∘H59",
    }.get(arm, arm)


def run_id(arm: str, seed: int) -> str:
    return f"{STUDENT}__{arm}__P0__S0__seed{seed}"


def parse_run_id(name: str) -> dict:
    student, arm, probe, seg, seed = name.split("__")
    return {"student": student, "arm": arm, "probe": probe, "seg": seg, "seed": int(seed.removeprefix("seed"))}


# Section 7 of the plan, written into decision.json before any training run
DECISION_RULE = {
    "delta": "Avg pass@1 (ALG) - Avg pass@1 (H59), percentage points",
    "threshold": "max(1.0, 2 sigma); sigma = max(std of uniform, std of ALG) of Avg pass@1 over seeds 42/43/44",
    "ci": "95% paired bootstrap over problems (10,000 resamples, seed 42, stratified by benchmark)",
    "rows": {
        "soft_better": "delta >= threshold and CI excludes 0: soft allocation beats hard selection; "
        "group B if there is time",
        "leans_soft": "delta >= 1.0 but not the above: run group C (H59 seeds 43, 44), then decide on 3-seed "
        "means with a Welch test",
        "equivalent": "|delta| < 1.0: run group C; if still equivalent the benefit comes from the signal, not "
        "from keeping every step",
        "hard_better": "delta <= -1.0: run group C and revisit the paper's main claim",
    },
    "secondary": [
        "pass@3: same rule; report both if it contradicts pass@1",
        "H59 - R59 >= 1.0: answer gain picks better steps than chance",
        "H59 below uniform: direct evidence for the cost of hard selection",
        "Q3: H59 NLL on its dropped tokens above uniform at epoch 3: the student forgets dropped steps",
        "no conclusion from a single benchmark",
    ],
}


def apply_rule(delta: float, threshold: float, ci: tuple[float, float] | None) -> str:
    excludes_zero = ci is not None and (ci[0] > 0 or ci[1] < 0)
    if delta >= threshold and excludes_zero:
        return "soft_better"
    if delta >= 1.0:
        return "leans_soft"
    if delta <= -1.0:
        return "hard_better"
    return "equivalent"


def needs_group_c(row: str | None) -> bool:
    return row in ("leans_soft", "equivalent", "hard_better")
