"""Shared constants and file helpers of the positional-bias experiment."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ALERD as published (arm e7-alg-tau1): lambda 0.5, tau 1, clip 2
INTERPOLATION, TEMPERATURE, CLIP, EPSILON = 0.5, 1.0, 2.0, 1e-8
N_DECILES = 10
MIN_STEPS = 5  # A2 / A3 only: a correlation over fewer steps means nothing
END_MARKER = "<End_of_Prefix>"
BENCHMARK_ORDER = ("aime25", "aime24", "amc12", "math500")

# Published reference numbers (percent). The decision rule is fixed against these before any run.
PUBLISHED = {
    "P-ALIGN (uniform)": {"aime25": (16.67, 26.67), "aime24": (17.78, 30.00), "amc12": (50.20, 63.86),
                          "math500": (78.40, 87.60), "avg": (40.76, 52.03)},
    "Random assignment (Table 3)": {"aime25": (13.33, 23.33), "aime24": (17.78, 23.33), "amc12": (53.41, 63.86),
                                    "math500": (77.87, 87.40), "avg": (40.60, 49.48)},
    "ALERD": {"aime25": (23.33, 33.33), "aime24": (17.78, 30.00), "amc12": (53.01, 67.47),
              "math500": (77.00, 88.00), "avg": (42.78, 54.70)},
}
TABLE8_WEIGHTS = {"w_p10": 0.595, "w_p50": 0.761, "w_p90": 1.233}


def decile(p: float) -> int:
    return min(N_DECILES - 1, int(N_DECILES * p))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    head = run("rev-parse", "HEAD") or "unknown"
    return f"{head}-dirty" if run("status", "--porcelain") else head


def provenance(inputs: dict | None = None) -> dict:
    """Who wrote a file, from what: goes into every JSON of the experiment (or meta/ for CSVs)."""
    return {
        "git_commit": git_commit(),
        "created_at": now(),
        "script": " ".join(sys.argv),
        "inputs_sha256": {name: sha256(path) for name, path in (inputs or {}).items() if path and Path(path).is_file()},
    }


def write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def save_figure(figure, path_without_suffix) -> None:
    import matplotlib.pyplot as plt

    Path(path_without_suffix).parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        figure.savefig(f"{path_without_suffix}{suffix}", bbox_inches="tight", dpi=200)
    plt.close(figure)
