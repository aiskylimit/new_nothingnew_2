"""Resolve experiment options and freeze them before launching GPU workers."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import yaml

from .cli import parser
from .config import load_config


def prepare(training_args, run_root, run_dir=None):
    args = parser().parse_args(training_args)
    cfg = load_config(
        args.config,
        {k: v for k, v in vars(args).items() if k not in {"config", "resume", "validate_only"}},
    ).validate(int(os.environ.get("NUM_PROCESSES", "4")))
    evaluation = {
        "steps": os.environ.get("EVAL_STEPS", "25,50,75"),
        "datasets": os.environ.get("EVAL_DATASETS", "aime24,aime25"),
        "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        "gpu_memory_utilization": float(os.environ.get("EVAL_GPU_MEMORY_UTILIZATION", "0.9")),
    }
    settings = asdict(cfg)
    # Output/cache locations are not experimental hyperparameters.
    fingerprint = {k: v for k, v in settings.items() if k not in {"output_dir", "replay_cache_dir"}}
    digest = hashlib.sha256(
        json.dumps({"config": fingerprint, "evaluation": evaluation}, sort_keys=True).encode()
    ).hexdigest()[:12]
    labels = (
        ("rho", cfg.ftrl_rho),
        ("rms", cfg.delta_rms_decay),
        ("floor", cfg.delta_rms_floor),
        ("clip", cfg.delta_scaled_clip),
        ("duel", cfg.control_batch_size),
        ("pool", cfg.control_pool_size),
        ("min", cfg.min_valid_controls),
        ("w", cfg.full_opsd_warmup_steps),
        ("H", cfg.window_steps),
        ("aw", cfg.warmup_windows),
        ("clr", cfg.controller_lr),
        ("tau", cfg.exposure_temperature),
        ("lgt", cfg.lambda_gt),
        ("gam", cfg.gamma_discount),
        ("b", cfg.effective_batch_size),
        ("lr", cfg.learning_rate),
        ("ans", int(cfg.mask_teacher_include_final_answer)),
        ("s", cfg.seed),
        ("es", cfg.eval_seed),
        ("n", cfg.total_updates),
    )
    name = f"{cfg.variant}_{cfg.teacher_selection}_" + "_".join(
        f"{label}{value:g}" for label, value in labels
    )
    name = name[:200].rstrip("_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(
        run_dir or args.output_dir or Path(run_root) / f"{name}_{digest}_{stamp}_{os.getpid()}"
    ).resolve()
    if any(
        (output / name).exists()
        for name in ("run_manifest.json", "metrics.jsonl", "train_eval.log", "config.yaml")
    ) or any(output.glob("checkpoint-*")):
        raise FileExistsError(f"RUN_DIR already contains a run: {output}")
    cfg.output_dir = str(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "config_hash": digest,
                "source_config": str(Path(args.config).resolve()),
                "evaluation": evaluation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main():
    setup = argparse.ArgumentParser(description=__doc__)
    setup.add_argument("--run-root", required=True)
    setup.add_argument("--run-dir")
    options, training_args = setup.parse_known_args()
    print(prepare(training_args, options.run_root, options.run_dir))


if __name__ == "__main__":
    main()
