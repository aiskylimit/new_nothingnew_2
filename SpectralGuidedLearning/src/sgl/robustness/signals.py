"""Per-step signals, weights and the training file of one W4 variant (probe x segmentation).

From the segmented traces and the raw scores of sgl.signals.answer_gain (one set, or the four probes
of Pavg, averaged per step) this writes signals/<student>__<probe>__<seg>/:

  steps.parquet   one row per step: boundaries (tokens and characters), text, region, b, u, z, v, clipped,
                  r, w_hat, w, answer flags, predictability
  traces.parquet  one row per trace: K, lengths, gold answer, mu/sigma of u, fallback, budget B, probe
                  context, top-1 step, positive gain in the top 3
  summary.json    counts, distributions, the paper's default-run statistics, every check
  manifest.json   git state, versions, hardware, input hashes, probe text and ids, segmentation

and the training file (loss_weights from IWC-Stable with tau=1, lambda=0.5, c=2: the published arm),
plus `token_tags` the step logger splits the unweighted NLL by. Exits non-zero if a check fails.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.allocation.iwc import stable_iwc_step_weights
from sgl.diagnostics.answer_steps import mention_spans
from sgl.eval.graders.answer_scoring import extract_boxed, normalize_math
from sgl.robustness import common as C
from sgl.segment.sentence import encode_with_offsets

# token_tags bits (sgl.positional.runlog.StepLogger splits the unweighted NLL by them)
TAG_PREFIX, TAG_CONT, TAG_JUNCTION, TAG_ANSWER_STEP, TAG_EOS, TAG_STEP = 1, 2, 4, 8, 16, 32


def iwc_parts(u: np.ndarray, lengths: np.ndarray):
    """z, v = clip(z), r, w_hat, w of Eqs. 4-6; w is taken from sgl.allocation.iwc (what trains)."""
    w = np.array(
        stable_iwc_step_weights(
            u.tolist(), [int(x) for x in lengths], C.TEMPERATURE, C.INTERPOLATION, C.CLIP, C.EPSILON
        )
    )
    std = u.std()
    fallback = bool(std < C.EPSILON)
    z = np.zeros_like(u) if fallback else (u - u.mean()) / (std + C.EPSILON)
    v = np.clip(z, -C.CLIP, C.CLIP)
    r = np.exp(v / C.TEMPERATURE)
    w_hat = r * lengths.sum() / (lengths * r).sum()
    if not np.allclose((1 - C.INTERPOLATION) + C.INTERPOLATION * w_hat, w, rtol=0, atol=1e-9):
        raise AssertionError("z/r/w_hat do not reproduce the IWC-Stable weights")
    return z, v, r, w_hat, w, fallback


def load_scores(paths: list[str]) -> list[dict]:
    return [C.read_json(p) if p and Path(p).is_file() else {} for p in paths]


def mean_lists(values: list[list[float] | None]) -> list[float] | None:
    present = [v for v in values if v is not None]
    if len(present) != len(values):
        return None  # a trace one probe could not score stays without a signal, as in the single-probe case
    return np.mean(np.array(present, dtype=float), axis=0).tolist()


def trace_rows(record, tokenizer, gains, logp, token_logp, longest_context, context_limit):
    response = record["response"]
    start, end = record["response_token_span"]
    ids, token_starts = encode_with_offsets(tokenizer, response)
    if ids != record["input_ids"][start:end]:
        raise ValueError(f"record {record['id']}: the tokenizer does not reproduce the stored response ids")
    spans = [(s["token_start"] - start, s["token_end"] - start) for s in record["steps"]]
    lengths = np.array([b - a for a, b in spans], dtype=float)
    K = len(spans)
    has_gain = gains is not None
    u = np.array(gains if has_gain else [0.0] * K, dtype=float)
    if len(u) != K:
        raise ValueError(f"record {record['id']}: {len(u)} gains for {K} steps")
    z, v, r, w_hat, w, fallback = iwc_parts(u, lengths)

    marker = response.find(C.END_MARKER)
    n_prefix = None
    if marker >= 0:
        boundary = marker + len(C.END_MARKER)
        n_prefix = sum(1 for s in token_starts if s < boundary)
    final_boxed = response.rfind("\\boxed{")
    gold = extract_boxed(response)
    gold_norm = normalize_math(gold) if gold else ""
    char_of = lambda t: token_starts[t] if t < len(token_starts) else len(response)  # noqa: E731

    rows = []
    for k, (first, last) in enumerate(spans):
        c0, c1 = char_of(first), char_of(last)
        text = response[c0:c1]
        if n_prefix is None:
            region = "no_marker"
        elif last <= n_prefix:
            region = "prefix"
        elif first >= n_prefix:
            region = "continuation"
        else:
            region = "crosses_marker"
        rows.append(
            {
                "trace_id": record["id"],
                "problem_id": record["id"],
                "k": k + 1,
                "tok_start": first,
                "tok_end": last,
                "ell": int(lengths[k]),
                "char_start": c0,
                "char_end": c1,
                "text": text,
                "region": region,
                "has_final_boxed": final_boxed >= 0 and c0 <= final_boxed < c1,
                "b": logp[k + 1] if logp else math.nan,
                "u": u[k],
                "z": z[k],
                "v": v[k],
                "clipped": bool(abs(z[k]) > C.CLIP),
                "r": r[k],
                "w_hat": w_hat[k],
                "w": w[k],
                "contains_answer_exact": bool(gold) and bool(mention_spans(text, gold)),
                "contains_answer_normalized": bool(gold_norm) and gold_norm in normalize_math(text),
                "predictability": float(token_logp[first:last].mean()) if token_logp is not None else math.nan,
            }
        )
    order = np.argsort(-u, kind="stable")
    positive = np.clip(u, 0, None)
    best = int(order[0])
    trace = {
        "trace_id": record["id"],
        "K": K,
        "num_response_tokens": end - start,
        "num_prefix_tokens": n_prefix if n_prefix is not None else -1,
        "gold_answer": gold,
        "gold_token_ids": tokenizer(gold + "}", add_special_tokens=False)["input_ids"] if gold else [],
        "gold_matches_reference": None,  # the 966-trace file carries no s1K-1.1 reference answer
        "has_gain": has_gain,
        "b0": logp[0] if logp else math.nan,
        "sum_u_minus_bK_b0": (float(u.sum()) - (logp[-1] - logp[0])) if logp and has_gain else math.nan,
        "mu_u": float(u.mean()),
        "sigma_u": float(u.std()),
        "uniform_fallback": fallback,
        "B": float((lengths * w).sum() / lengths.sum()),
        "max_probe_context_len": longest_context,
        "exceeds_context": bool(longest_context and longest_context >= context_limit),
        "top1_step": best + 1,
        "top1_contains_answer": rows[best]["contains_answer_exact"],
        "top3_pos_gain_share": float(positive[order[:3]].sum() / positive.sum()) if positive.sum() > 0 else math.nan,
    }
    trace["N"] = len(trace["gold_token_ids"])
    return rows, trace, n_prefix


def training_record(record, table: pd.DataFrame, n_prefix, column: str | None):
    """Same supervision as sgl.allocation.build (every response token + the stop token, weight 1 on the
    stop token); `column` None = uniform weights."""
    start, end = record["response_token_span"]
    n = len(record["input_ids"])
    mask, weights, source, tags = [0] * n, [0.0] * n, [0] * n, [0] * n
    spans = []
    for row in table.itertuples():
        first, last = int(row.tok_start) + start, int(row.tok_end) + start
        weight = 1.0 if column is None else float(getattr(row, column))
        spans.append([first, last, weight])
        answer = TAG_ANSWER_STEP if row.contains_answer_exact else 0
        for position in range(first, last):
            mask[position], weights[position] = 1, weight
            relative = position - start
            if n_prefix is not None and relative < n_prefix:
                source[position], tag = 1, TAG_PREFIX
            else:
                source[position], tag = 2, TAG_CONT
                if n_prefix is not None and relative < n_prefix + C.JUNCTION_TOKENS:
                    tag |= TAG_JUNCTION
            tags[position] = tag | answer | TAG_STEP
    for position in range(end, n):  # the stop token
        mask[position], weights[position], source[position], tags[position] = 1, 1.0, 2, TAG_EOS
    return {
        "id": record["id"],
        "input_ids": record["input_ids"],
        "loss_mask": mask,
        "loss_weights": weights,
        "token_source": source,
        "token_tags": tags,
        "step_spans_w": spans,
    }


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarize(steps: pd.DataFrame, traces: pd.DataFrame) -> dict:
    K = traces.K.to_numpy()
    gained = steps[traces.set_index("trace_id").has_gain.reindex(steps.trace_id).to_numpy()]
    positive = gained.u.clip(lower=0)
    mass = steps.ell * steps.w
    top = steps.w >= steps.w.quantile(0.9)
    answer = gained.contains_answer_exact
    return {
        "n_traces": len(traces),
        "n_steps": len(steps),
        "n_probes": int(len(steps) + len(traces)),
        "steps_per_trace": {
            "min": int(K.min()),
            "p25": float(np.percentile(K, 25)),
            "median": float(np.median(K)),
            "p75": float(np.percentile(K, 75)),
            "max": int(K.max()),
        },
        "n_single_step_traces": int((K == 1).sum()),
        "n_traces_without_gain": int((~traces.has_gain).sum()),
        "pct_traces_uniform_fallback": float(traces.uniform_fallback.mean()),
        "pct_steps_clipped": float(steps.clipped.mean()),
        "top10pct_mass_share": float(mass[top].sum() / mass.sum()),
        "top10pct_mass_share_definition": (
            "share of corpus token-weight mass (ell*w) held by the 10% of steps with the largest w"
        ),
        "steps_up_0.1nat": float((gained.u > 0.1).mean()),
        "steps_down_0.1nat": float((gained.u < -0.1).mean()),
        "pos_gain_share_answer_steps": float(positive[answer].sum() / positive.sum()) if positive.sum() else math.nan,
        "pos_gain_share_answer_steps_definition": (
            "sum of positive u on steps containing the gold answer / sum of positive u"
        ),
        "pct_positive_steps_with_answer": float(answer[gained.u > 0].mean()) if (gained.u > 0).any() else math.nan,
        "traces_top_step_has_answer": float(traces[traces.has_gain].top1_contains_answer.mean()),
        "w_quantiles": {q: float(steps.w.quantile(float(q))) for q in ("0.1", "0.5", "0.9", "0.99")},
        "u_by_region_median": {k: float(v) for k, v in gained.groupby("region").u.median().items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="segmented traces of this variant's segmentation")
    parser.add_argument("--probe", required=True, help="P0..P3, Pavg")
    parser.add_argument("--seg", required=True, choices=sorted(C.SEGMENTATIONS))
    parser.add_argument("--gains", nargs="+", required=True, help="answer_gain output(s); several are averaged (Pavg)")
    parser.add_argument("--logp", nargs="*", default=[], help="answer_gain --logp-output, same order")
    parser.add_argument("--meta", nargs="*", default=[], help="answer_gain --meta-output, same order")
    parser.add_argument("--token-logp", help="sgl.robustness.token_logp npz (predictability column)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-output", required=True, help="training JSONL of this variant")
    parser.add_argument("--uniform-output", help="also write the uniform-weight training JSONL (same tags)")
    parser.add_argument("--reference-gains", help="P0/S0 only: the gains ALERD was trained on (regression)")
    parser.add_argument("--reference-dataset", help="P0/S0 only: ALERD's training JSONL (weights must match)")
    parser.add_argument("--hashes", help="meta/hashes.json (traces file, tokenizer, base model)")
    parser.add_argument("--gpu-mem-file", help="peak GPU memory sampled during scoring (MiB, one per line)")
    parser.add_argument("--context-limit", type=int, default=16384)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = [json.loads(line) for line in open(args.data_path)]
    gains_sets, logp_sets, metas = load_scores(args.gains), load_scores(args.logp), load_scores(args.meta)
    token_logp = np.load(args.token_logp) if args.token_logp and Path(args.token_logp).is_file() else None

    rows, traces, prefixes = [], [], {}
    for record in records:
        key = str(record["id"])
        gains = mean_lists([g.get(key) for g in gains_sets])
        logp = mean_lists([lp.get(key) for lp in logp_sets]) if logp_sets and all(logp_sets) else None
        longest = max((m.get("max_probe_context_len", {}).get(key, 0) for m in metas), default=0)
        step_rows, trace, n_prefix = trace_rows(
            record,
            tokenizer,
            gains,
            logp,
            token_logp[key] if token_logp is not None else None,
            longest,
            args.context_limit,
        )
        rows += step_rows
        traces.append(trace)
        prefixes[record["id"]] = n_prefix
    steps, traces = pd.DataFrame(rows), pd.DataFrame(traces)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps.to_parquet(out / "steps.parquet", index=False)
    traces.to_parquet(out / "traces.parquet", index=False)

    summary, failures = summarize(steps, traces), []
    lengths_ok = (
        steps.groupby("trace_id").ell.sum().reindex(traces.trace_id).to_numpy() == traces.num_response_tokens.to_numpy()
    )
    budget = float((traces.B - 1).abs().max())
    checks = {
        "token_coverage_ok": bool(lengths_ok.all()),
        "max_abs_budget_minus_1": budget,
        "max_abs_sum_u_minus_bK_b0": float(traces.sum_u_minus_bK_b0.abs().max()) if logp_sets else None,
        "n_traces_exceeding_context": int(traces.exceeds_context.sum()),
    }
    if not lengths_ok.all():
        failures.append("sum of step lengths != response tokens")
    if budget > 1e-6:
        failures.append("budget invariance sum(ell*w) = sum(ell) violated")
    if args.reference_gains and Path(args.reference_gains).is_file():
        reference = C.read_json(args.reference_gains)
        ours = gains_sets[0]
        common_ids = [k for k in ours if k in reference]
        diffs = [
            max(abs(a - b) for a, b in zip(ours[k], reference[k]))
            for k in common_ids
            if len(ours[k]) == len(reference[k])
        ]
        rhos = [
            spearman(ours[k], reference[k])
            for k in common_ids
            if len(ours[k]) == len(reference[k]) and len(ours[k]) > 2
        ]
        checks["regression_vs_reference_gains"] = {
            "n_traces": len(common_ids),
            "n_step_count_mismatch": len(common_ids) - len(diffs),
            "max_abs_diff": max(diffs) if diffs else None,
            "median_abs_diff_per_trace": float(np.median(diffs)) if diffs else None,
            "median_within_trace_spearman": float(np.nanmedian(rhos)) if rhos else None,
        }
        if len(diffs) != len(common_ids) or not common_ids:
            failures.append("regression: step counts differ from the reference gains")
    if args.reference_dataset and Path(args.reference_dataset).is_file():
        by_trace = {i: g.sort_values("k") for i, g in steps.groupby("trace_id")}
        worst = 0.0
        with open(args.reference_dataset) as handle:
            for line in handle:
                ref = json.loads(line)
                record = next(r for r in records if r["id"] == ref["id"])
                ours = training_record(record, by_trace[ref["id"]], prefixes[ref["id"]], "w")["loss_weights"]
                worst = max(
                    worst,
                    max((abs(a - b) for a, b in zip(ours, ref["loss_weights"])), default=0.0)
                    if len(ours) == len(ref["loss_weights"])
                    else math.inf,
                )
        checks["max_abs_weight_diff_vs_reference_dataset"] = worst
    if args.probe == "P0" and args.seg == "S0":
        checks["default_reference"] = {k: {"paper": v, "ours": summary.get(k)} for k, v in C.DEFAULT_REFERENCE.items()}
    peak = None
    if args.gpu_mem_file and Path(args.gpu_mem_file).is_file():
        values = [float(x) for x in Path(args.gpu_mem_file).read_text().split() if x.strip()]
        peak = max(values) / 1024 if values else None
    summary.update(
        {
            "checks": checks,
            "failures": failures,
            "scoring_seconds": sum(m.get("seconds", 0) for m in metas) or None,
            "peak_gpu_mem_gb": peak,
        }
    )
    summary["provenance"] = C.provenance()
    C.write_json(out / "summary.json", summary)

    kind, parameter = C.SEGMENTATIONS[args.seg]
    C.write_json(
        out / "manifest.json",
        {
            "variant": {"probe": args.probe, "seg": args.seg, "student": C.STUDENT},
            "probe": [
                {"text": m.get("probe"), "token_ids": m.get("probe_token_ids"), "tokens": m.get("probe_tokens")}
                for m in metas
            ]
            or [{"text": C.PROBES.get(args.probe)}],
            "probe_members": list(C.PAVG_MEMBERS) if args.probe == "Pavg" else [args.probe],
            "segmentation": {
                "code": args.seg,
                "kind": kind,
                "parameter": parameter,
                "function": "sgl.robustness.segment.chunk_spans"
                if kind == "chunk"
                else f"sgl.segment.SEGMENTERS[{parameter!r}]",
            },
            "weights": {
                "tau": C.TEMPERATURE,
                "lambda": C.INTERPOLATION,
                "c": C.CLIP,
                "epsilon": C.EPSILON,
                "function": "sgl.allocation.iwc.stable_iwc_step_weights",
            },
            "inputs": {
                "data_path": args.data_path,
                "gains": args.gains,
                "logp": args.logp,
                "meta": args.meta,
                "token_logp": args.token_logp,
            },
            "inputs_sha256": {
                p: C.sha256(p) for p in [args.data_path, *args.gains, *args.logp] if p and Path(p).is_file()
            },
            "hashes": C.read_json(args.hashes) if args.hashes and Path(args.hashes).is_file() else None,
            "steps_parquet_sha256": C.sha256(out / "steps.parquet"),
            "dataset_output": args.dataset_output,
            "git": C.git_state(),
            "versions": C.versions(),
            "hardware": C.hardware(),
            "scoring_seconds": summary["scoring_seconds"],
            "created_at": C.now(),
        },
    )
    if failures:
        raise SystemExit(f"{out.name}: checks FAILED, do not train:\n  " + "\n  ".join(failures))

    by_trace = {i: g.sort_values("k") for i, g in steps.groupby("trace_id")}
    outputs = [(args.dataset_output, "w")] + ([(args.uniform_output, None)] if args.uniform_output else [])
    for path, column in outputs:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            for record in records:
                handle.write(
                    json.dumps(training_record(record, by_trace[record["id"]], prefixes[record["id"]], column)) + "\n"
                )
        print(f"training file -> {path}")
    print(
        f"{out.name}: {len(traces)} traces, {len(steps):,} steps; positive gain on answer steps "
        f"{summary['pos_gain_share_answer_steps']:.1%}, top step has answer {summary['traces_top_step_has_answer']:.1%}"
    )


if __name__ == "__main__":
    main()
