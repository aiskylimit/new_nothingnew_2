"""Variants, run names and shared numbers of the W4 experiment (probe and segmentation robustness).

Layout under results/w4/ (the plan's section 5.1):
  signals/<student>__<probe>__<seg>/   manifest.json, steps.parquet, traces.parquet, summary.json, raw/
  runs/<run_id>/                       manifest.json, config.yaml, train/, checkpoints/, eval/, raw/
  analysis/                            rank_stability/, rank_stability_summary.csv, results_master.csv
"""

import importlib.metadata
import platform
import subprocess

from sgl.positional.common import (  # noqa: F401  (re-exported)
    BENCHMARK_ORDER,
    END_MARKER,
    now,
    provenance,
    read_json,
    sha256,
    write_json,
)

STUDENT = "r1q1.5b"
# ALERD as published (e7-alg-tau1): every W4 variant keeps them, they were chosen on P0 + S0
TEMPERATURE, INTERPOLATION, CLIP, EPSILON = 1.0, 0.5, 2.0, 1e-8
HYPER = "tau1_lam0.5_c2"
JUNCTION_TOKENS = 8  # first continuation tokens after <End_of_Prefix>

# Every probe has an explicit "answer" cue and ends with "\boxed{", so the gold answer (with its "}")
# keeps one format and only the wording changes.
PROBES = {
    "P0": "\n\nTherefore, the final answer is \\boxed{",
    "P1": "\n\nThe answer is \\boxed{",
    "P2": "\n\nFinal Answer: \\boxed{",
    "P3": "\n\n**Final Answer**\n\n\\boxed{",
    "P3alt": "\n\nThe final result is $\\boxed{",  # fallback if P3 turns out too close to P2
}
PAVG_MEMBERS = ("P0", "P1", "P2", "P3")

# seg code -> (kind, parameter): "rule" = a lossless text rule of sgl.segment, "chunk" = fixed token blocks
SEGMENTATIONS = {
    "S0": ("rule", "sentence"),
    "S1": ("rule", "paragraph"),
    "S2a": ("chunk", 64),
    "S2b": ("chunk", 128),
    "S3": ("rule", "math_sentence"),
}

# (probe, seg) pairs that need scoring; Pavg is the mean of PAVG_MEMBERS on S0
SCORED = [
    ("P0", "S0"),
    ("P1", "S0"),
    ("P2", "S0"),
    ("P3", "S0"),
    ("P0", "S1"),
    ("P0", "S2a"),
    ("P0", "S2b"),
    ("P0", "S3"),
]
VARIANTS = SCORED + [("Pavg", "S0")]

# training priorities of the plan (section 3.2); C = seed noise
PRIORITY = {
    "A": [("alg", "P1", "S0"), ("alg", "Pavg", "S0"), ("alg", "P0", "S1"), ("alg", "P0", "S3")],
    "B": [("alg", "P2", "S0"), ("alg", "P3", "S0"), ("alg", "P0", "S2a"), ("alg", "P0", "S2b")],
    "C": [("uniform", "none", "none"), ("alg", "P0", "S0")],
}
NOISE_SEEDS = (43, 44)  # seed 42 is the existing E0 / E7 run

# Default (P0, S0) numbers quoted in the paper, checked by the regression step
DEFAULT_REFERENCE = {
    "n_steps": 47532,
    "n_probes": 48498,
    "pos_gain_share_answer_steps": 0.283,
    "traces_top_step_has_answer": 0.66,
    "top10pct_mass_share": 0.395,
    "steps_up_0.1nat": 0.43,
    "steps_down_0.1nat": 0.39,
}
PUBLISHED = {
    "Uniform (P-ALIGN)": {
        "aime25": (16.67, 26.67),
        "aime24": (17.78, 30.00),
        "amc12": (50.20, 63.86),
        "math500": (78.40, 87.60),
        "avg": (40.76, 52.03),
    },
    "ALG default (P0, S0)": {
        "aime25": (23.33, 33.33),
        "aime24": (17.78, 30.00),
        "amc12": (53.01, 67.47),
        "math500": (77.00, 88.00),
        "avg": (42.78, 54.70),
    },
}
UNIFORM_AVG = PUBLISHED["Uniform (P-ALIGN)"]["avg"]
N_EVAL_ROWS = (30 + 30 + 83 + 500) * 3  # samples.jsonl lines of a complete evaluation


def signal_name(probe: str, seg: str) -> str:
    return f"{STUDENT}__{probe}__{seg}"


def dataset_variant(method: str, probe: str, seg: str) -> str:
    """train_variant of the training file data/<track>/train-<variant>.jsonl."""
    return "w4-uniform" if method == "uniform" else f"w4-{probe}-{seg}"


def run_id(method: str, probe: str, seg: str, seed: int) -> str:
    if method == "uniform":
        probe = seg = "none"
    return f"{STUDENT}__{method}__{probe}__{seg}__{HYPER}__seed{seed}"


def parse_run_id(name: str) -> dict:
    student, method, probe, seg, hyper, seed = name.split("__")
    return {
        "student": student,
        "method": method,
        "probe": probe,
        "seg": seg,
        "hyper": hyper,
        "seed": int(seed.removeprefix("seed")),
    }


def versions() -> dict:
    out = {}
    for package in ("torch", "transformers", "deepspeed", "vllm", "peft", "accelerate", "math-verify"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    try:
        import torch

        out["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return out


def hardware() -> dict:
    info = {"hostname": platform.node()}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        info["gpus"] = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = None
    return info


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True).stdout
        except OSError:  # no git binary on this machine
            return ""

    diff = run("diff", "HEAD")
    return {
        "commit": run("rev-parse", "HEAD").strip() or "unknown",
        "dirty": bool(run("status", "--porcelain").strip()),
        "diff": diff or None,
    }
