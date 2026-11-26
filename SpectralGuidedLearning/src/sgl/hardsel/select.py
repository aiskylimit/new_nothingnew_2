"""Choose the kept steps of every arm, check them and write the training files (CPU).

Rule H_p: rank a trace's steps by the unclipped gain u (ties: earlier step first) and add whole steps until
the kept tokens reach the budget (p x response tokens, or the SGL spectral gate's kept tokens of that
trace). R: the same budget as H59 filled in a random step order. The stop token is always supervised;
dropped tokens stay in the input and only leave the loss. alg-h59 = H59's mask with ALG's weights
(tau 1, lambda 0.5, c 2) recomputed over the kept steps.

Every arm is checked first; if any check fails no training file is written and the exit code is non-zero.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sgl.allocation.build import build_example
from sgl.allocation.iwc import stable_iwc_step_weights
from sgl.hardsel import common as C
from sgl.robustness import common as W4
from sgl.robustness.signals import TAG_PREFIX, trace_rows, training_record

TOLERANCE = 1e-6


def alg_weights(u, lengths) -> list[float]:
    return stable_iwc_step_weights(
        [float(x) for x in u], [int(x) for x in lengths], W4.TEMPERATURE, W4.INTERPOLATION, W4.CLIP, W4.EPSILON
    )


def rank_by_gain(u) -> list[int]:
    return sorted(range(len(u)), key=lambda k: (-u[k], k))


def fill(order: list[int], lengths, target: float) -> list[bool]:
    """Whole steps in `order` until the kept tokens reach `target` (at least one step)."""
    kept, total = [False] * len(lengths), 0
    for k in order:
        kept[k] = True
        total += int(lengths[k])
        if total >= target - 1e-9:
            break
    return kept


def load_gains(args, records) -> tuple[dict[int, list[float] | None], dict]:
    """{trace id: per-step u, or None without a gain} and where it came from."""
    if args.gains and Path(args.gains).is_file():
        raw = C.read_json(args.gains)
        return {r["id"]: raw.get(str(r["id"])) for r in records}, {"source": "gains_json", "path": args.gains}
    if not args.w4_signals or not Path(args.w4_signals, "steps.parquet").is_file():
        raise SystemExit(
            f"no answer gain: neither {args.gains} nor {args.w4_signals}/steps.parquet exists "
            "(run 'bash project_commands_w4.sh score' first)"
        )
    directory = Path(args.w4_signals)
    summary = C.read_json(directory / "summary.json")
    if summary.get("failures"):
        raise SystemExit(f"{directory}: the P0/S0 checks failed ({summary['failures']}), its gains are not used")
    steps = pd.read_parquet(directory / "steps.parquet").sort_values(["trace_id", "k"])
    has_gain = pd.read_parquet(directory / "traces.parquet").set_index("trace_id").has_gain
    gains = {int(i): (g.u.tolist() if has_gain[i] else None) for i, g in steps.groupby("trace_id")}
    return gains, {
        "source": "w4_rescored_P0_S0",
        "path": str(directory / "steps.parquet"),
        "regression_vs_reference_gains": summary.get("checks", {}).get("regression_vs_reference_gains"),
    }


def sgl_kept_tokens(path: str, records) -> dict[int, tuple[int, list[bool]]]:
    """{id: (kept tokens, per-step kept)} of the SGL spectral gate's training file (train-spectral.jsonl)."""
    masks = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            masks[row["id"]] = (row["input_ids"], row["loss_mask"])
    out = {}
    for record in records:
        if record["id"] not in masks or masks[record["id"]][0] != record["input_ids"]:
            raise SystemExit(f"{path}: trace {record['id']} is missing or tokenized differently")
        mask = masks[record["id"]][1]
        kept = [bool(mask[s["token_start"]]) for s in record["steps"]]
        for s, k in zip(record["steps"], kept):
            if any(bool(x) != k for x in mask[s["token_start"] : s["token_end"]]):
                raise SystemExit(f"{path}: trace {record['id']} does not mask whole steps of this segmentation")
        out[record["id"]] = (sum(s["token_end"] - s["token_start"] for s, k in zip(record["steps"], kept) if k), kept)
    return out


def build_tables(records, gains, tokenizer):
    """Per-trace step tables (regions, answer flags, ...) from sgl.robustness.signals, plus the gain rank."""
    tables, meta = {}, {}
    for record in records:
        g = gains.get(record["id"])
        rows, trace, n_prefix = trace_rows(record, tokenizer, g, None, None, 0, 16384)
        table = pd.DataFrame(rows)
        order = rank_by_gain(table.u.tolist())
        rank = np.empty(len(table), dtype=int)
        rank[order] = np.arange(1, len(table) + 1)
        table["rank_in_trace"] = rank
        length = trace["num_response_tokens"]
        table["rel_pos"] = (table.tok_start + table.ell / 2) / length
        tables[record["id"]] = table
        meta[record["id"]] = {"n_prefix": n_prefix, "L": length, "has_gain": trace["has_gain"]}
    return tables, meta


def choose(records, tables, meta, budgets: dict | None):
    """{arm: {id: per-step kept}}, {arm: {id: per-step weight}}, {arm: {id: target tokens}}."""
    kept = {arm: {} for arm in C.ARMS}
    weights = {arm: {} for arm in C.ARMS}
    target = {arm: {} for arm in C.ARMS}
    for record in records:
        i = record["id"]
        table, length = tables[i], meta[i]["L"]
        u, ell = table.u.tolist(), table.ell.tolist()
        order = rank_by_gain(u)
        target["h59"][i] = budgets[i][0] if budgets else C.FIXED_FRACTION["h59"] * length
        target["h30"][i] = C.FIXED_FRACTION["h30"] * length
        kept["h59"][i] = fill(order, ell, target["h59"][i])
        kept["h30"][i] = fill(order, ell, target["h30"][i])
        kept["alg-h59"][i] = list(kept["h59"][i])
        target["alg-h59"][i] = target["h59"][i]
        target["r59"][i] = sum(n for n, k in zip(ell, kept["h59"][i]) if k)
        shuffled = list(range(len(ell)))
        random.Random(C.RANDOM_SEED + i).shuffle(shuffled)
        kept["r59"][i] = fill(shuffled, ell, target["r59"][i])
        for arm in C.ARMS:
            if arm == "alg-h59":
                chosen = [k for k, flag in enumerate(kept[arm][i]) if flag]
                values = alg_weights([u[k] for k in chosen], [ell[k] for k in chosen])
                full = [0.0] * len(ell)
                for k, value in zip(chosen, values):
                    full[k] = value
                weights[arm][i] = full
            else:
                weights[arm][i] = [float(flag) for flag in kept[arm][i]]
    return kept, weights, target


def to_training_record(record, table, n_prefix, kept, weights, h59_kept) -> dict:
    out = training_record(record, table, n_prefix, None)
    start = record["response_token_span"][0]
    for row, keep, weight, h59 in zip(table.itertuples(), kept, weights, h59_kept):
        first, last = int(row.tok_start) + start, int(row.tok_end) + start
        bits = (C.TAG_KEPT if keep else C.TAG_DROPPED) | (0 if h59 else C.TAG_H59_DROPPED)
        out["step_spans_w"][int(row.k) - 1][2] = weight
        for position in range(first, last):
            out["loss_mask"][position] = int(keep)
            out["loss_weights"][position] = weight
            out["token_tags"][position] |= bits
    return out


def check_arm(arm, records, tables, meta, kept, weights, target, h59_kept, sgl_budget: bool) -> dict:
    failures, excess_over_h59, overshoot = [], [], []
    mass_error, min_weight = 0.0, math.inf
    for record in records:
        i = record["id"]
        table, length = tables[i], meta[i]["L"]
        u, ell = table.u.to_numpy(), table.ell.to_numpy()
        flags, w = np.array(kept[arm][i]), np.array(weights[arm][i])
        n_kept = int(ell[flags].sum())
        fail = lambda text, i=i: failures.append(f"trace {i}: {text}")  # noqa: E731
        if not flags.any():
            fail("no step kept")
        if int(ell.sum()) != length or len(record["input_ids"]) - record["response_token_span"][0] != length + 1:
            fail("sum of step lengths + stop token != response tokens")
        if n_kept < target[arm][i] - 1e-9 or n_kept > target[arm][i] + ell.max() + 1e-9:
            fail(f"kept {n_kept} tokens outside [{target[arm][i]:.1f}, +longest step]")
        overshoot.append((n_kept - target[arm][i]) / length)
        if arm in ("h59", "h30", "alg-h59") and (~flags).any() and u[flags].min() < u[~flags].max() - 1e-12:
            fail("a dropped step has a higher gain than a kept one")
        if arm == "r59":
            h59_tokens = int(ell[np.array(h59_kept[i])].sum())
            excess_over_h59.append(n_kept - h59_tokens)
            if not 0 <= n_kept - h59_tokens <= ell.max():
                fail(f"kept {n_kept} tokens, H59 kept {h59_tokens}")
        if (w[~flags] != 0).any():
            fail("a dropped step has weight")
        if arm == "alg-h59":
            error = abs(float((ell * w).sum()) / n_kept - 1)
            mass_error = max(mass_error, error)
            min_weight = min(min_weight, float(w[flags].min()))
            if error > TOLERANCE or w[flags].min() < 1 - W4.INTERPOLATION - TOLERANCE:
                fail("ALG weights do not keep the kept token mass / drop below 1 - lambda")
        elif not np.isin(w, (0.0, 1.0)).all() or int((ell * w).sum()) != n_kept:
            fail("weights are not binary")
    report = {"arm": arm, "n_traces": len(records), "n_failures": len(failures), "failures": failures[:20]}
    report["max_kept_frac_over_target"] = float(max(overshoot))
    if sgl_budget and arm != "h30":
        report["budget"] = "per-trace kept tokens of the SGL spectral gate"
    if excess_over_h59:
        q = np.quantile(excess_over_h59, [0, 0.25, 0.5, 0.75, 1])
        report["kept_tokens_minus_h59"] = dict(zip(("min", "p25", "median", "p75", "max"), map(float, q)))
    if arm == "alg-h59":
        report |= {"max_abs(sum_kept_ell_w/sum_kept_ell - 1)": mass_error, "min_kept_weight": min_weight}
    return report


def check_alg_function(records, tables, meta, reference_dataset: str | None, gain_source: str) -> tuple[dict, list]:
    """With every step kept, alg-h59's weight function must give E7's training file (original gains) or
    sgl.allocation.build's output for the same gains."""
    reference = {}
    if reference_dataset and Path(reference_dataset).is_file():
        with open(reference_dataset) as handle:
            reference = {row["id"]: row for row in map(json.loads, handle)}
    worst_function = worst_file = 0.0
    for record in records:
        table = tables[record["id"]]
        w = alg_weights(table.u, table.ell)
        everything = [True] * len(w)
        ours = to_training_record(record, table, meta[record["id"]]["n_prefix"], everything, w, everything)
        built, _ = build_example(
            record, [1.0] * len(w), table.u.tolist(), 1.0, "iwc-stable",
            W4.TEMPERATURE, W4.INTERPOLATION, W4.CLIP, W4.EPSILON,
        )
        pairs = zip(ours["loss_weights"], built["loss_weights"])
        worst_function = max(worst_function, max(abs(a - b) for a, b in pairs))
        if reference:
            other = reference.get(record["id"], {}).get("loss_weights")
            if other is None or len(other) != len(ours["loss_weights"]):
                worst_file = math.inf
            else:
                worst_file = max(worst_file, max(abs(a - b) for a, b in zip(ours["loss_weights"], other)))
    failures = []
    if worst_function > TOLERANCE:
        failures.append(f"all-ones mask differs from sgl.allocation.build by {worst_function:.2e}")
    # E7's file was built from the original gains: only then must it be reproduced exactly
    if reference and gain_source == "gains_json" and worst_file > TOLERANCE:
        failures.append(f"all-ones mask differs from {reference_dataset} by {worst_file:.2e}")
    report = {
        "max_abs_diff_vs_allocation_build": worst_function,
        "reference_dataset": reference_dataset if reference else None,
        "max_abs_diff_vs_reference_dataset": worst_file if reference else None,
        "reference_dataset_enforced": bool(reference) and gain_source == "gains_json",
    }
    return report, failures


def top_gain_flags(u) -> np.ndarray:
    flags = np.zeros(len(u), dtype=bool)
    flags[rank_by_gain(list(u))[: math.ceil(0.1 * len(u))]] = True
    return flags


def coefficient_shares(records, tables, weights) -> dict:
    """Per-trace share of the loss coefficients (step tokens) held by each region, averaged over traces."""
    sums = {k: [] for k in ("prefix", "continuation", "answer_steps", "top10pct_gain_steps")}
    for record in records:
        table = tables[record["id"]]
        ell, w = table.ell.to_numpy(), np.array(weights[record["id"]])
        mass = ell * w
        total = mass.sum()
        n_prefix = table.attrs["n_prefix"]
        prefix_tokens = np.zeros(len(table)) if n_prefix is None else np.clip(n_prefix - table.tok_start, 0, ell)
        prefix = float((prefix_tokens * w).sum() / total)
        sums["prefix"].append(prefix)
        sums["continuation"].append(1 - prefix)
        sums["answer_steps"].append(float(mass[table.contains_answer_exact.to_numpy()].sum() / total))
        sums["top10pct_gain_steps"].append(float(mass[top_gain_flags(table.u)].sum() / total))
    return {k: float(np.mean(v)) for k, v in sums.items()}


def arm_stats(records, tables, meta, kept, weights) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    step_frames, trace_rows_ = [], []
    for record in records:
        i = record["id"]
        table, info = tables[i], meta[i]
        flags, ell = np.array(kept[i]), table.ell.to_numpy()
        n_prefix = info["n_prefix"]
        prefix_tokens = np.zeros(len(table)) if n_prefix is None else np.clip(n_prefix - table.tok_start, 0, ell)
        prefix_total = float(prefix_tokens.sum())
        prefix_kept = float(prefix_tokens[flags].sum())
        cont_total = info["L"] - prefix_total
        answer = table.contains_answer_exact.to_numpy()
        boxed = table.has_final_boxed.to_numpy()
        trace_rows_.append({
            "trace_id": i, "K": len(table), "L": info["L"], "has_gain": info["has_gain"],
            "kept_tokens": int(ell[flags].sum()), "kept_frac": float(ell[flags].sum() / info["L"]),
            "n_kept_steps": int(flags.sum()), "final_boxed_kept": bool(flags[boxed].any()) if boxed.any() else None,
            "n_answer_steps": int(answer.sum()), "n_answer_steps_kept": int((answer & flags).sum()),
            "n_prefix_tokens": prefix_total, "kept_prefix_tokens": prefix_kept,
            "prefix_kept_frac": prefix_kept / prefix_total if prefix_total else math.nan,
            "cont_kept_frac": (int(ell[flags].sum()) - prefix_kept) / cont_total if cont_total else math.nan,
        })
        step_frames.append(table.assign(kept=flags, w=weights[i]))
    steps, traces = pd.concat(step_frames, ignore_index=True), pd.DataFrame(trace_rows_)
    prefix = traces.n_prefix_tokens.sum()
    cont = traces.L.sum() - prefix
    cont_kept = traces.kept_tokens.sum() - traces.kept_prefix_tokens.sum()
    decile = np.minimum((steps.rel_pos * 10).astype(int), 9)
    by_decile = (steps.ell * steps.kept).groupby(decile).sum() / steps.ell.groupby(decile).sum()
    with_boxed = traces.final_boxed_kept.dropna()
    stats = {
        "n_traces": len(traces),
        "n_traces_without_gain": int((~traces.has_gain).sum()),
        "kept_step_frac": float(steps.kept.mean()),
        "kept_token_frac": float(traces.kept_tokens.sum() / traces.L.sum()),
        "kept_token_frac_trace_mean": float(traces.kept_frac.mean()),
        "kept_token_frac_prefix": float(traces.kept_prefix_tokens.sum() / prefix) if prefix else None,
        "kept_token_frac_continuation": float(cont_kept / cont) if cont else None,
        "pct_traces_keeping_final_boxed": float(with_boxed.astype(bool).mean()) if len(with_boxed) else None,
        "pct_answer_steps_kept": float(steps.kept[steps.contains_answer_exact].mean())
        if steps.contains_answer_exact.any()
        else None,
        "kept_token_frac_by_position_decile": [float(by_decile.get(d, math.nan)) for d in range(10)],
    }
    return stats, steps, traces


def sgl_overlap(records, tables, kept, budgets) -> dict:
    jaccard_steps, jaccard_tokens = [], []
    for record in records:
        i = record["id"]
        ell = tables[i].ell.to_numpy()
        a, b = np.array(kept[i]), np.array(budgets[i][1])
        union = a | b
        jaccard_steps.append(float((a & b).sum() / union.sum()) if union.any() else 1.0)
        jaccard_tokens.append(float(ell[a & b].sum() / ell[union].sum()) if union.any() else 1.0)
    return {"median_jaccard_steps": float(np.median(jaccard_steps)),
            "median_jaccard_tokens": float(np.median(jaccard_tokens))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl (S0)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True, help="where train-hard-<arm>.jsonl go")
    parser.add_argument("--root", default="results/hard")
    parser.add_argument("--gains", help="the gains E7 was trained on (answer_gain.json), used if present")
    parser.add_argument("--w4-signals", help="fallback: results/w4/signals/r1q1.5b__P0__S0")
    parser.add_argument("--reference-dataset", help="train-e7-alg-tau1.jsonl, if kept")
    parser.add_argument("--sgl-mask", help="train-spectral.jsonl (SGL gate), used for the Jaccard and the sgl budget")
    parser.add_argument(
        "--budget", choices=("auto", "fixed", "sgl"), default="auto",
        help="H59/R59 budget: fixed 59%%, or the SGL gate's kept tokens per trace (H-SGL); "
        "auto = sgl if --sgl-mask exists",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    records = [json.loads(line) for line in open(args.data_path)]
    gains, gain_info = load_gains(args, records)
    tables, meta = build_tables(records, gains, tokenizer)
    for i, table in tables.items():
        table.attrs["n_prefix"] = meta[i]["n_prefix"]

    have_mask = bool(args.sgl_mask) and Path(args.sgl_mask).is_file()
    if args.budget == "sgl" and not have_mask:
        raise SystemExit(f"--budget sgl needs the SGL gate file, none at {args.sgl_mask}")
    gate = sgl_kept_tokens(args.sgl_mask, records) if have_mask else None
    budget_source = "sgl_gate" if gate and args.budget != "fixed" else "fixed"
    kept, weights, target = choose(records, tables, meta, gate if budget_source == "sgl_gate" else None)

    root = Path(args.root)
    alg_report, failures = check_alg_function(records, tables, meta, args.reference_dataset, gain_info["source"])
    uniform_w = {r["id"]: [1.0] * len(tables[r["id"]]) for r in records}
    alg_w = {r["id"]: alg_weights(tables[r["id"]].u, tables[r["id"]].ell) for r in records}
    reference_shares = {"uniform": coefficient_shares(records, tables, uniform_w),
                        "alg": coefficient_shares(records, tables, alg_w)}
    checks = {}
    for arm in C.ARMS:
        report = check_arm(arm, records, tables, meta, kept, weights, target, kept["h59"],
                           budget_source == "sgl_gate")
        if arm == "alg-h59":
            report["all_ones_mask"] = alg_report
            report["failures"] += failures
            report["n_failures"] += len(failures)
        checks[arm] = report
        directory = root / "selection" / arm
        C.write_json(directory / "select_check.json", {**report, "provenance": C.provenance()})
        stats, steps, traces = arm_stats(records, tables, meta, kept[arm], weights[arm])
        stats["loss_coefficient_share"] = coefficient_shares(records, tables, weights[arm])
        stats["loss_coefficient_share_reference"] = reference_shares
        if gate:
            stats["overlap_with_sgl_gate"] = sgl_overlap(records, tables, kept[arm], gate)
        C.write_json(directory / "selection_stats.json",
                     {**stats, "arm": arm, "label": C.label(arm, budget_source), "budget_source": budget_source,
                      "share_definition": "per trace: coefficients of the region / all coefficients (step tokens, "
                      "stop token excluded), averaged over traces; top10pct = the ceil(10%) highest-u steps of a trace",
                      "provenance": C.provenance()})
        columns = ["trace_id", "k", "tok_start", "tok_end", "ell", "u", "z", "rank_in_trace", "kept", "w", "region",
                   "rel_pos", "contains_answer_exact", "contains_answer_normalized", "has_final_boxed"]
        steps[columns].to_parquet(directory / "steps.parquet", index=False)
        traces.drop(columns=["n_prefix_tokens", "kept_prefix_tokens"]).to_parquet(
            directory / "traces.parquet", index=False)

    if gate:
        gate_kept = {i: value[1] for i, value in gate.items()}
        gate_stats, _, _ = arm_stats(records, tables, meta, gate_kept, {i: [float(x) for x in v]
                                                                        for i, v in gate_kept.items()})
        C.write_json(root / "analysis/selection_overlap.json", {
            "sgl_gate_file": args.sgl_mask,
            "sgl_gate": gate_stats,
            **{arm: sgl_overlap(records, tables, kept[arm], gate) for arm in ("h59", "h30", "r59")},
            "note": "with the sgl budget, h59 and the gate differ only in the step score" if budget_source == "sgl_gate"
            else "fixed budget: token totals of h59 and the gate differ",
            "provenance": C.provenance(),
        })

    bad = {arm: c["failures"] for arm, c in checks.items() if c["n_failures"]}
    if bad:
        raise SystemExit("selection checks FAILED, no training file written:\n" + json.dumps(bad, indent=1)[:3000])

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for arm in C.ARMS:
        path = data_dir / f"train-hard-{arm}.jsonl"
        with path.open("w") as handle:
            for record in records:
                i = record["id"]
                row = to_training_record(record, tables[i], meta[i]["n_prefix"], kept[arm][i], weights[arm][i],
                                         kept["h59"][i])
                handle.write(json.dumps(row) + "\n")
        files[arm] = str(path)
    # the manifest is written last: its presence means every training file is complete
    for arm in C.ARMS:
        C.write_json(root / "selection" / arm / "manifest.json", {
            "arm": arm, "label": C.label(arm, budget_source), "budget_source": budget_source,
            "target_fraction": C.FIXED_FRACTION.get("h30" if arm == "h30" else "h59")
            if arm == "h30" or budget_source == "fixed" else None,
            "r59_shuffle_seed": f"random.Random({C.RANDOM_SEED} + trace id)",
            "gain": gain_info, "gain_sha256": C.sha256(gain_info["path"]),
            "segmented_file": args.data_path, "segmented_sha256": C.sha256(args.data_path),
            "tokenizer": args.tokenizer,
            "sgl_mask": args.sgl_mask if gate else None, "sgl_mask_sha256": C.sha256(args.sgl_mask) if gate else None,
            "training_file": files[arm], "training_file_sha256": C.sha256(files[arm]),
            "steps_parquet_sha256": C.sha256(root / "selection" / arm / "steps.parquet"),
            "n_token_tags_bits": {"prefix": TAG_PREFIX, "kept": C.TAG_KEPT, "dropped": C.TAG_DROPPED,
                                  "h59_dropped": C.TAG_H59_DROPPED},
            "git": C.git_state(), "created_at": C.now(),
        })
    print(f"gain: {gain_info['source']} ({gain_info['path']}); H59/R59 budget: {budget_source}")
    for arm in C.ARMS:
        stats = C.read_json(root / "selection" / arm / "selection_stats.json")
        print(f"{C.label(arm, budget_source):10s} kept {stats['kept_token_frac']:.1%} of tokens, "
              f"{stats['kept_step_frac']:.1%} of steps -> {files[arm]}")


if __name__ == "__main__":
    main()
