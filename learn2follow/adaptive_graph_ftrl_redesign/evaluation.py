"""Evaluate checkpoints through the repository's existing math evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .eval_adapter import run_eval, shell_eval_line


def main(default_launcher="python"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", default="25,50,75,100")
    parser.add_argument("--datasets", default="aime25")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model")
    parser.add_argument("--dataset-dir", default=os.environ.get("EVAL_DATASET_DIR"))
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--launcher", choices=["python", "se-opsd"], default=default_launcher)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.run_dir.strip():
        parser.error("--run-dir must be a nonempty checkpoint run directory")
    manifest = Path(args.run_dir) / "run_manifest.json"
    config = json.loads(manifest.read_text()).get("config", {}) if manifest.is_file() else {}
    settings = {
        "gpus": args.gpus,
        "launcher": args.launcher,
        "dataset_dir": args.dataset_dir,
        "seed": args.seed if args.seed is not None else config.get("eval_seed", 42),
        "model": args.model or config.get("model", "Qwen/Qwen3-4B"),
        "max_new_tokens": args.max_new_tokens or config.get("eval_max_new_tokens", 38912),
        "temperature": config.get("eval_temperature", 1.0),
        "top_p": config.get("eval_top_p", 0.95),
        "top_k": config.get("eval_top_k", -1),
        "samples": config.get("eval_samples", 12),
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else config.get("eval_gpu_memory_utilization", 0.9)
        ),
    }
    if not 0 < settings["gpu_memory_utilization"] <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not args.gpus.strip() or any(not gpu.strip() for gpu in args.gpus.split(",")):
        parser.error("--gpus must contain a nonempty comma-separated device list")
    print(
        json.dumps(
            {
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "top_k": settings["top_k"],
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "samples_per_prompt": settings["samples"],
                "thinking": True,
                "seed": settings["seed"],
                "max_new_tokens": settings["max_new_tokens"],
                "gpus": settings["gpus"],
                "gpu_memory_utilization": settings["gpu_memory_utilization"],
                "launcher": settings["launcher"],
                "metric": f"Avg@{settings['samples']} (not Pass@{settings['samples']})",
            }
        ),
        flush=True,
    )

    for step in [s.strip() for s in args.steps.split(",") if s.strip()]:
        ckpt = str(Path(args.run_dir) / f"checkpoint-{step}")
        for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            if args.dry_run:
                print(
                    shell_eval_line(
                        ckpt,
                        dataset,
                        **settings,
                    )
                )
            else:
                if not (Path(ckpt) / "adapter_model.safetensors").is_file():
                    raise FileNotFoundError(ckpt)
                code = run_eval(
                    ckpt,
                    dataset,
                    **settings,
                )
                if code:
                    raise SystemExit(code)


if __name__ == "__main__":
    main()
