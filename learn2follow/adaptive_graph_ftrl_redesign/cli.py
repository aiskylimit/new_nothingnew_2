"""Training arguments shared by the trainer and lightweight experiment setup."""

import argparse


def parser():
    p = argparse.ArgumentParser(description="Train Adaptive Graph FTRL Redesign.")
    p.add_argument("--config", default="adaptive_graph_ftrl_redesign/configs/graph_answer.yaml")
    for name in ("dataset", "output-dir", "model", "replay-cache-dir"):
        p.add_argument("--" + name)
    p.add_argument("--variant", choices=["full", "prefix", "graph"])
    p.add_argument("--teacher-selection", choices=["soft", "hard"])
    for name in (
        "total-updates",
        "save-every",
        "seed",
        "eval-seed",
        "data-limit",
        "full-opsd-warmup-steps",
        "window-steps",
        "warmup-windows",
        "max-pair-candidates",
        "gradient-accumulation-steps",
        "effective-batch-size",
        "control-batch-size",
        "control-pool-size",
        "min-valid-controls",
        "grounded-score-examples-per-step",
        "grounded-reference-max-tokens",
    ):
        p.add_argument("--" + name, type=int)
    for name in (
        "lambda-gt",
        "gamma-discount",
        "ftrl-rho",
        "ftrl-eta",
        "ftrl-min-weight",
        "delta-rms-decay",
        "delta-rms-floor",
        "delta-scaled-clip",
        "normalizer-decay",
        "reward-variance-floor",
        "delta-variance-floor",
        "standardizer-clip",
        "controller-lr",
        "controller-max-grad-norm",
        "exposure-temperature",
        "answer-temperature",
        "learning-rate",
        "adam-beta1",
        "adam-beta2",
        "adam-epsilon",
        "weight-decay",
        "max-grad-norm",
        "vllm-gpu-memory-utilization",
    ):
        p.add_argument("--" + name, type=float)
    p.add_argument("--generation-backend", choices=["vllm", "transformers"])
    for name in (
        "mask-teacher-include-final-answer",
        "frozen-controller",
        "fixed-mixture",
        "verbose-runtime",
    ):
        p.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--resume")
    p.add_argument("--validate-only", action="store_true")
    return p
