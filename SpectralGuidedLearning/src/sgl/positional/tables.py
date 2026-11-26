"""Step/trace tables, the position-only weights R6-R8, their sanity checks and training datasets.

  tables/step_table.parquet, trace_table.parquet   source of every analysis
  weights/w_r6|w_r7|w_r8.parquet (+ .sha256)       per-step weights of the control runs
  analysis/sanity_weights.json                     exits non-zero when a requirement fails
  <data-output-dir>/train-positional-r6|r7|r8.jsonl  datasets for sgl.training.train

Schemes: r6 = ALERD's own pipeline with the score s_k = p_k (relative step position);
r7 = ALERD's corpus mean weight per position decile, re-normalized per trace (no step content);
r8 = ALERD's z-scores permuted within each position decile across the corpus, then clip/exp/normalize.
Token positions are relative to the response; tok_end is exclusive.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.allocation.iwc import stable_iwc_step_weights
from sgl.diagnostics.answer_steps import mention_spans
from sgl.eval.graders.answer_scoring import extract_boxed
from sgl.positional import common as C
from sgl.transforms.provenance import prefix_token_count

SCHEMES = ("alerd", "r6", "r7", "r8")


def stable_parts(z: np.ndarray, lengths: np.ndarray):
    """Eqs. 4-5 from z-scores: r = exp(clip(z)/tau), mass-preserving w_hat, shrunk w."""
    r = np.exp(np.clip(z, -C.CLIP, C.CLIP) / C.TEMPERATURE)
    w_hat = r * lengths.sum() / (lengths * r).sum()
    return r, w_hat, (1 - C.INTERPOLATION) + C.INTERPOLATION * w_hat


def standardize(values: np.ndarray):
    std = values.std()
    if std < C.EPSILON:
        return np.zeros_like(values), True
    return (values - values.mean()) / (std + C.EPSILON), False


def trace_rows(record: dict, gains: list[float] | None, logp: list[float] | None, tokenizer):
    start, end = record["response_token_span"]
    spans = [(s["token_start"] - start, s["token_end"] - start) for s in record["steps"]]
    lengths = np.array([e - s for s, e in spans], dtype=float)
    n_steps, total = len(spans), lengths.sum()
    u = np.array(gains if gains is not None else [0.0] * n_steps, dtype=float)
    if len(u) != n_steps:
        raise ValueError(f"record {record['id']}: {len(u)} gains for {n_steps} steps")
    p = (np.cumsum(lengths) - lengths / 2) / total
    z, sigma_zero = standardize(u)
    r, w_hat, w = stable_parts(z, lengths)
    # the weights the published arm trains on come from sgl.allocation.iwc: take exactly those
    exact = np.array(stable_iwc_step_weights(
        u.tolist(), [int(x) for x in lengths], C.TEMPERATURE, C.INTERPOLATION, C.CLIP, C.EPSILON))
    if not np.allclose(w, exact, rtol=0, atol=1e-9):
        raise AssertionError(f"record {record['id']}: z/r/w_hat do not reproduce the IWC-Stable weights")
    w_r6 = stable_iwc_step_weights(
        p.tolist(), [int(x) for x in lengths], C.TEMPERATURE, C.INTERPOLATION, C.CLIP, C.EPSILON)

    ids = record["input_ids"][start:end]
    n_prefix = prefix_token_count(tokenizer, record["response"], ids)
    answer = extract_boxed(record["response"])
    rows = []
    for k, (first, last) in enumerate(spans):
        text = tokenizer.decode(ids[first:last])
        in_prefix = max(0, min(last, n_prefix) - first) if n_prefix is not None else 0
        rows.append({
            "trace_id": record["id"], "k": k + 1, "K": n_steps, "tok_start": first, "tok_end": last,
            "ell": int(lengths[k]), "p": p[k], "decile": C.decile(p[k]),
            "b": logp[k + 1] if logp else math.nan, "u": u[k], "z": z[k], "r": r[k], "w_hat": w_hat[k],
            "w_alerd": exact[k], "n_tok_prefix": in_prefix,
            "n_tok_cont": (last - first - in_prefix) if n_prefix is not None else 0,
            "has_marker": n_prefix is not None and first < n_prefix <= last,
            "contains_answer": bool(answer) and bool(mention_spans(text, answer)),
            "is_last": k + 1 == n_steps, "w_r6": w_r6[k], "text_head": text[:300],
        })
    best = int(np.argmax(u))
    trace = {
        "trace_id": record["id"], "K": n_steps, "L": int(total), "marker_tok": n_prefix - 1 if n_prefix else -1,
        "n_tok_prefix": n_prefix if n_prefix is not None else -1, "sigma_u_zero": sigma_zero,
        "has_gain": gains is not None, "excluded_A2A3": n_steps < C.MIN_STEPS, "argmax_k": best + 1,
        "argmax_p": p[best], "argmax_is_last": best + 1 == n_steps,
        "argmax_contains_answer": rows[best]["contains_answer"],
    }
    return rows, trace


def add_r7(steps: pd.DataFrame) -> pd.DataFrame:
    """Corpus mean ALERD weight per decile (token-weighted), then per-trace mass preservation."""
    mass = (steps.ell * steps.w_alerd).groupby(steps.decile).sum()
    profile = (mass / steps.ell.groupby(steps.decile).sum()).reindex(range(C.N_DECILES)).fillna(1.0)
    raw = steps.decile.map(profile)
    scale = steps.groupby("trace_id").ell.transform("sum") / (steps.ell * raw).groupby(steps.trace_id).transform("sum")
    steps["w_r7"] = raw * scale
    return pd.DataFrame({"decile": range(C.N_DECILES), "w_bar": profile.values})


def add_r8(steps: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    z = steps.z.to_numpy()
    permuted = z.copy()
    counts = {}
    for d in range(C.N_DECILES):
        index = np.flatnonzero(steps.decile.to_numpy() == d)
        permuted[index] = z[rng.permutation(index)]
        counts[d] = len(index)
    steps["z_r8"] = permuted
    weights = np.empty(len(steps))
    for index in steps.groupby("trace_id").indices.values():
        weights[index] = stable_parts(permuted[index], steps.ell.to_numpy(dtype=float)[index])[2]
    steps["w_r8"] = weights
    return {"seed": seed, "steps_per_decile": counts}


def weight_summary(steps: pd.DataFrame, column: str) -> dict:
    weights = steps[column].to_numpy()
    token_mass = (steps.ell * steps[column]).groupby(steps.trace_id).sum()
    ratio = token_mass / steps.ell.groupby(steps.trace_id).sum()
    # step-level quantiles, as Table 8 (sgl.allocation.build.summarize) computes them
    return {
        "max_abs_mass_ratio_minus_1": float((ratio - 1).abs().max()),
        "min_w": float(weights.min()), "w_max": float(weights.max()),
        "w_p10": float(np.quantile(weights, 0.1)), "w_p50": float(np.quantile(weights, 0.5)),
        "w_p90": float(np.quantile(weights, 0.9)), "w_p99": float(np.quantile(weights, 0.99)),
        "w_mean_token": float((steps.ell * steps[column]).sum() / steps.ell.sum()),
    }


def profile_R(steps: pd.DataFrame, column: str) -> np.ndarray:
    mass = (steps.ell * steps[column]).groupby(steps.decile).sum().reindex(range(C.N_DECILES)).fillna(0)
    uniform = steps.ell.groupby(steps.decile).sum().reindex(range(C.N_DECILES)).fillna(0)
    return ((mass / mass.sum()) / (uniform / uniform.sum())).to_numpy()


def sanity_checks(steps: pd.DataFrame, traces: pd.DataFrame, dataset_diff: float | None) -> tuple[dict, list[str]]:
    report, failures = {}, []

    def require(ok: bool, message: str) -> None:
        if not ok:
            failures.append(message)

    sigma_zero = int(traces.sigma_u_zero.sum())
    for scheme in SCHEMES:
        item = weight_summary(steps, f"w_{scheme}")
        item["n_traces_sigma_zero"] = sigma_zero
        require(item["max_abs_mass_ratio_minus_1"] < 1e-6, f"{scheme}: token mass not preserved per trace")
        if scheme != "r7":
            require(item["min_w"] >= 1 - C.INTERPOLATION - 1e-9, f"{scheme}: a weight is below 1 - lambda")
        report[scheme] = item

    report["alerd"]["table8"] = C.TABLE8_WEIGHTS
    report["alerd"]["matches_table8"] = all(
        abs(report["alerd"][k] - v) < 0.002 for k, v in C.TABLE8_WEIGHTS.items())
    report["alerd"]["max_abs_diff_vs_training_dataset"] = dataset_diff
    require(dataset_diff is None or dataset_diff < 1e-6,
            "alerd: recomputed weights differ from the dataset ALERD was trained on")

    drops = steps.sort_values(["trace_id", "k"]).groupby("trace_id").w_r6.diff()
    report["r6"]["r6_monotonic_violations"] = int((drops < -1e-12).sum())
    require(report["r6"]["r6_monotonic_violations"] == 0, "r6: weights decrease along a trace")

    report["r7"]["r7_profile_max_abs_err"] = float(np.abs(profile_R(steps, "w_r7") - profile_R(steps, "w_alerd")).max())

    preserved = all(np.allclose(np.sort(g.z), np.sort(g.z_r8)) for _, g in steps.groupby("decile"))
    # Permuting inside a decile keeps the between-decile relation of z and position on purpose, so
    # "no correlation with the original z" can only hold within deciles.
    centred = lambda c: steps[c] - steps.groupby("decile")[c].transform("mean")  # noqa: E731
    a, b = centred("z"), centred("z_r8")
    within = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    raw = float(np.corrcoef(steps.z, steps.z_r8)[0, 1]) if steps.z.std() > 0 else 0.0
    limit = max(0.05, 4 / math.sqrt(len(steps)))
    report["r8"].update({"r8_multiset_preserved": bool(preserved), "r8_corr_with_original_z": within,
                         "r8_corr_with_original_z_raw": raw, "r8_corr_limit": limit})
    require(preserved, "r8: z multiset changed inside a decile")
    require(abs(within) < limit, "r8: permuted z still correlates with the original z within deciles")
    return report, failures


def training_records(records: list[dict], steps: pd.DataFrame, column: str):
    by_trace = {i: g.sort_values("k") for i, g in steps.groupby("trace_id")}
    for record in records:
        start, end = record["response_token_span"]
        table = by_trace[record["id"]]
        n = len(record["input_ids"])
        mask, weights, source = [0] * n, [0.0] * n, [0] * n
        n_prefix = int(table.n_tok_prefix.sum())
        spans = []
        for first, last, weight in zip(table.tok_start, table.tok_end, table[column]):
            first, last, weight = int(first) + start, int(last) + start, float(weight)
            spans.append([first, last, weight])
            mask[first:last] = [1] * (last - first)
            weights[first:last] = [weight] * (last - first)
        for position in range(start, end):
            if mask[position]:
                source[position] = 1 if position < start + n_prefix else 2
        for position in range(end, n):  # the stop token: weight 1, as in every other arm
            mask[position], weights[position], source[position] = 1, 1.0, 2
        yield {"id": record["id"], "input_ids": record["input_ids"], "loss_mask": mask,
               "loss_weights": weights, "token_source": source, "step_spans_w": spans}


def dataset_max_diff(path: str, records: list[dict], steps: pd.DataFrame) -> float:
    """Largest |recomputed - trained| ALERD token weight against the arm's own dataset."""
    expected = {r["id"]: r["loss_weights"] for r in training_records(records, steps, "w_alerd")}
    worst, seen = 0.0, 0
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            ours = expected[row["id"]]
            if len(ours) != len(row["loss_weights"]):
                return math.inf
            worst = max(worst, max(abs(a - b) for a, b in zip(ours, row["loss_weights"])))
            seen += 1
    return worst if seen == len(expected) else math.inf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--gains", required=True, help="the answer-gain scores ALERD was built from")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default="results/positional")
    parser.add_argument("--data-output-dir", help="where the training JSONL files go (default: next to the data)")
    parser.add_argument("--logp", help="optional per-step log p(answer) (answer_gain --logp-output), column b")
    parser.add_argument("--alerd-dataset", help="optional ALERD training JSONL to check the weights against")
    parser.add_argument("--seed", type=int, default=42, help="R8 permutation")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with open(args.data_path) as handle:
        records = [json.loads(line) for line in handle]
    gains = C.read_json(args.gains)
    logp = C.read_json(args.logp) if args.logp and Path(args.logp).is_file() else {}

    rows, traces = [], []
    for record in records:
        key = str(record["id"])
        step_rows, trace = trace_rows(record, gains.get(key), logp.get(key), tokenizer)
        rows += step_rows
        traces.append(trace)
    steps, traces = pd.DataFrame(rows), pd.DataFrame(traces)
    r7_profile = add_r7(steps)
    r8_info = add_r8(steps, args.seed)
    for scheme in SCHEMES:
        mass = (steps.ell * steps[f"w_{scheme}"]).groupby(steps.trace_id).sum()
        traces[f"mass_ratio_{scheme}"] = traces.trace_id.map(mass) / traces.L

    out = Path(args.output_dir)
    for sub in ("meta", "tables", "analysis", "weights"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    steps.to_parquet(out / "tables/step_table.parquet", index=False)
    traces.to_parquet(out / "tables/trace_table.parquet", index=False)
    r7_profile.to_csv(out / "weights/r7_profile.csv", index=False)
    C.write_json(out / "weights/r8_perm.json", r8_info)
    for scheme in ("r6", "r7", "r8"):
        path = out / f"weights/w_{scheme}.parquet"
        steps[["trace_id", "k", "tok_start", "tok_end", f"w_{scheme}"]].rename(
            columns={f"w_{scheme}": "w"}).to_parquet(path, index=False)
        Path(f"{path}.sha256").write_text(f"{C.sha256(path)}  {path.name}\n")

    inputs = {"dataset": args.data_path, "gains": args.gains, "logp": args.logp, "alerd_dataset": args.alerd_dataset}
    diff = None
    if args.alerd_dataset and Path(args.alerd_dataset).is_file():
        diff = dataset_max_diff(args.alerd_dataset, records, steps)
    report, failures = sanity_checks(steps, traces, diff)
    report["provenance"] = C.provenance(inputs)
    report["failures"] = failures
    C.write_json(out / "analysis/sanity_weights.json", report)
    C.write_json(out / "meta/tables.json", {
        **C.provenance(inputs), "n_traces": len(traces), "n_steps": len(steps),
        "n_traces_without_gain": int((~traces.has_gain).sum()),
        "n_traces_without_marker": int((traces.n_tok_prefix < 0).sum()),
        "has_b": bool(logp), "lambda": C.INTERPOLATION, "tau": C.TEMPERATURE, "clip": C.CLIP,
        "contains_answer": "sgl.diagnostics.answer_steps.mention_spans on the last \\boxed answer",
        "positions": "relative to the response, tok_end exclusive",
    })
    if not report["alerd"]["matches_table8"]:
        print("WARNING: ALERD weight percentiles differ from Table 8 (0.595/0.761/1.233): "
              f"{ {k: round(report['alerd'][k], 3) for k in C.TABLE8_WEIGHTS} } -- are these the published gains?")
    if failures:
        raise SystemExit("sanity checks FAILED, do not train:\n  " + "\n  ".join(failures))

    data_out = Path(args.data_output_dir or Path(args.data_path).parent)
    data_out.mkdir(parents=True, exist_ok=True)
    for scheme in ("r6", "r7", "r8"):
        path = data_out / f"train-positional-{scheme}.jsonl"
        with path.open("w") as handle:
            for row in training_records(records, steps, f"w_{scheme}"):
                handle.write(json.dumps(row) + "\n")
        print(f"{scheme}: {len(records)} records -> {path}")
    print(f"{len(traces)} traces, {len(steps)} steps -> {out}; sanity checks passed")


if __name__ == "__main__":
    main()
