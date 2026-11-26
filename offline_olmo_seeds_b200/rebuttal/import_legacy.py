"""Import results produced by the ORIGINAL per-method scripts (the ones behind
the paper's main table) into runs/<model>/<method>/seed<k>/, so analyze.py can
aggregate them over seeds.

The original scripts write, in their --output-dir:
    <tag>[_step<N>][_<benchmarks>]_generations.json   {bench: {problem_ids, golds, generations[k]}}
    <tag>_train_logs.json                              per-step training records (step is 0-based)
Every sample is re-graded with tropic.eval.grade (the same grader the scripts
use), so per-problem correctness is available for the bootstrap.

    PYTHONPATH=. python rebuttal/import_legacy.py --src results_tropic_g_8b_tokeff --tag tropic_g_8b \\
        --model qwen3-8b --method tropic --seed 0 --tokenizer Qwen/Qwen3-8B

Files without a _step<N> part are the end-of-training evaluation; give their
step with --default-step (100 for the 100-step scripts). Shard files
(_shard<i>of<n>) are ignored: merge them with scripts/combine_shards.py first.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebuttal.eval import answer_entropy  # noqa: E402
from rebuttal.presets import EVAL_MAX_NEW_TOKENS, MODEL_PRESETS, run_dir  # noqa: E402


def _grade_problem(job):
    texts, gold = job
    from tropic.eval import extract_boxed_answer, grade

    out = []
    for t in texts:
        a = extract_boxed_answer(t)
        out.append((bool(grade(t, gold)), a))
    return out


def find_generation_files(src: Path, tag: str, default_step: int | None) -> dict[int, list[Path]]:
    pat = re.compile(rf"^{re.escape(tag)}(?:_step(\d+))?(?:_[A-Za-z0-9\-]+)?_generations\.json$")
    by_step: dict[int, list[Path]] = {}
    for f in sorted(src.glob(f"{tag}*_generations.json")):
        if "_shard" in f.name:
            continue
        m = pat.match(f.name)
        if not m:
            continue
        if m.group(1) is not None:
            step = int(m.group(1))
        elif default_step is not None:
            step = default_step
        else:
            print(f"  skip {f.name}: no _step<N> in the name (pass --default-step)")
            continue
        by_step.setdefault(step, []).append(f)
    return by_step


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="--output-dir of the original script")
    ap.add_argument("--tag", required=True, help="file prefix, e.g. tropic_g_8b, opsd, rlsd, sdpo")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_PRESETS))
    ap.add_argument("--method", required=True, help="tropic / opsd / rlsd / sdpo (or <method>-<variant>)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--benchmarks", nargs="*", default=None, help="only these benchmarks (default: all found)")
    ap.add_argument("--steps", type=int, nargs="*", default=None, help="only these checkpoint steps")
    ap.add_argument("--default-step", type=int, default=None)
    ap.add_argument("--tokenizer", default=None, help="HF name/path to count tokens per sample (optional)")
    ap.add_argument("--max-new-tokens", type=int, default=EVAL_MAX_NEW_TOKENS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.src)
    method_name, _, variant = args.method.partition("-")
    rd = run_dir(args.runs_root, args.model, method_name, args.seed, variant or None)
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    by_step = find_generation_files(src, args.tag, args.default_step)
    if args.steps:
        by_step = {s: v for s, v in by_step.items() if s in set(args.steps)}
    if not by_step:
        raise SystemExit(f"no {args.tag}*_generations.json files in {src}")

    n_written = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for step, files in sorted(by_step.items()):
            merged = {}
            for f in files:
                merged.update(json.loads(f.read_text()))
            for bench, data in sorted(merged.items()):
                if args.benchmarks and bench not in args.benchmarks:
                    continue
                dest = rd / "eval" / f"step{step}" / f"{bench}.json"
                if dest.exists() and not args.overwrite:
                    print(f"  exists, skip: {dest}")
                    continue
                gens, golds, pids = data["generations"], data["golds"], data["problem_ids"]
                graded = list(pool.map(_grade_problem, zip(gens, golds)))
                rows = []
                for pid, gold, texts, g in zip(pids, golds, gens, graded):
                    samples = []
                    for t, (ok, ans) in zip(texts, g):
                        n_tok = len(tok(t, add_special_tokens=False)["input_ids"]) if tok is not None else None
                        samples.append(dict(correct=ok, n_tokens=n_tok, finish_reason=None,
                                            truncated=(n_tok >= args.max_new_tokens) if n_tok is not None else None,
                                            has_boxed=ans is not None, answer=ans))
                    rows.append(dict(problem_id=str(pid), gold=gold, samples=samples,
                                     answer_entropy=answer_entropy([s["answer"] for s in samples])))
                k = len(rows[0]["samples"]) if rows else 0
                avg = sum(s["correct"] for r in rows for s in r["samples"]) / max(1, len(rows) * k)
                dest.parent.mkdir(parents=True, exist_ok=True)
                meta = dict(model=args.model, method=args.method, seed=args.seed, step=step, benchmark=bench, k=k,
                            avg_at_k=avg, imported_from=str(files[0].name))
                dest.write_text(json.dumps(dict(meta=meta, problems=rows)))
                n_written += 1
                print(f"  step {step:>3} {bench:<7} avg@{k}={avg:.4f}  -> {dest}")

    log = src / f"{args.tag}_train_logs.json"
    if log.exists():
        recs = json.loads(log.read_text())
        lines = []
        for r in recs:
            r = dict(r)
            r["step"] = int(r["step"]) + 1          # original logs are 0-based; checkpoint N = after step N
            r["seed"] = args.seed
            if "kl_student_old_drift" in r and "kl_step_last" not in r:
                r["kl_step_last"] = r["kl_student_old_drift"]
            lines.append(json.dumps(r))
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "train_log.jsonl").write_text("\n".join(lines) + "\n")
        print(f"  train log: {len(recs)} steps -> {rd / 'train_log.jsonl'}")
    else:
        print(f"  (no {log.name}; cost/dynamics tables will skip this run)")
    print(f"done: {n_written} (step, benchmark) files written under {rd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
