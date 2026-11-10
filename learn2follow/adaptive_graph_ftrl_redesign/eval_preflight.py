"""Check checkpoints and writable storage before graph-based evaluation."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def check_storage(paths, min_free_gib=10.0, min_free_inodes=10000):
    if not math.isfinite(min_free_gib) or min_free_gib < 0 or min_free_inodes < 0:
        raise ValueError("Storage limits must be finite and nonnegative")
    for name, directory in paths.items():
        path = Path(directory).resolve()
        path.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free / 1024**3
        stats = os.statvfs(path)
        print(f"{name}: {path} | free={free:.1f} GiB | available_inodes={stats.f_favail}", flush=True)
        if free < min_free_gib:
            raise RuntimeError(
                f"{name} has only {free:.1f} GiB free; require {min_free_gib:g} GiB. "
                "Set EVAL_WORK_DIR/RESULTS_DIR to a filesystem with enough space."
            )
        if stats.f_files and stats.f_favail < min_free_inodes:
            raise RuntimeError(f"{name} has insufficient free inodes; select another filesystem")
        # A real write catches unwritable directories and some quota failures.
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b"evaluation storage probe\n")
            probe.flush()
            os.fsync(probe.fileno())


def check_checkpoints(run_dir, steps):
    from safetensors import safe_open

    if not steps or any(not step.isdecimal() or int(step) <= 0 for step in steps):
        raise ValueError("EVAL_STEPS must be comma-separated positive checkpoint steps")
    for step in steps:
        checkpoint = Path(run_dir) / f"checkpoint-{step}"
        config = json.loads((checkpoint / "adapter_config.json").read_text())
        rank = max([config["r"], *config.get("rank_pattern", {}).values()])
        if rank <= 0 or rank > 64:
            raise ValueError(
                f"Unsupported adapter rank {rank} in {checkpoint}; graph evaluator supports <=64"
            )
        with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
            if not tensors.keys():
                raise ValueError(f"Empty LoRA adapter: {checkpoint}")
        print(f"Checkpoint OK: {checkpoint}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", default="25,50")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--min-free-inodes", type=int, default=10000)
    args = parser.parse_args()
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(gpus) != 4 or any(not g.strip() for g in gpus) or len(set(g.strip() for g in gpus)) != 4:
        raise ValueError("This graph-based script requires four distinct GPU IDs")
    check_storage(
        {
            **{
                key: os.environ[key]
                for key in ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT")
            },
            "results": args.results_dir,
        },
        args.min_free_gib,
        args.min_free_inodes,
    )
    check_checkpoints(args.run_dir, args.steps.split(","))
    print("Preflight OK; starting requested evaluation without a compiler probe", flush=True)


if __name__ == "__main__":
    main()
