"""Table of the run 1 ALERD sweep (project_commands_tune_run1.sh): test scores of the ALERD configurations and uniform SFT.

No configuration is picked here: with no dev set, choosing by these scores would be selection on test.
Reads files only.
"""

import json
from pathlib import Path

TEST = ["aime24", "aime25", "math500", "amc12"]
OLD, NEW = "rebuttal/results", "rebuttal/tune/results"
ARMS = [  # tau, lambda, arm name, test results root
    ("-", "unif", "sft-nll", NEW),
    ("1", "0.5", "iwc-gain-l05", OLD),
    ("0.5", "0.5", "iwc-gain-t05-l05", NEW),
    ("2", "0.5", "iwc-gain-t2-l05", NEW),
    ("1", "0.3", "iwc-gain-t1-l03", NEW),
    ("2", "1.0", "iwc-gain-t2-l10", NEW),
]


def read(root: str, tag: str) -> dict:
    path = Path(root, tag, "summary.json")
    return {r["benchmark"]: r for r in json.loads(path.read_text())} if path.is_file() else {}


def cell(scores: dict, benchmarks: list[str]) -> str:
    if any(b not in scores for b in benchmarks):
        return "-"
    return " / ".join(
        f"{100 * sum(scores[b][k] for b in benchmarks) / len(benchmarks):.2f}" for k in ("pass@1", "pass@3")
    )


def main() -> None:
    print("pass@1 / pass@3 (%); test avg = mean of the four benchmarks")
    columns = [*TEST, "test avg"]
    print(f"{'tau':>4} {'lambda':>6}" + "".join(f"  {c:>14}" for c in columns))
    for tau, lam, arm, root in ARMS:
        test = read(root, f"{arm}-r1-qwen-1.5b-s1k")
        cells = [*(cell(test, [b]) for b in TEST), cell(test, TEST)]
        print(f"{tau:>4} {lam:>6}" + "".join(f"  {c:>14}" for c in cells))


if __name__ == "__main__":
    main()
