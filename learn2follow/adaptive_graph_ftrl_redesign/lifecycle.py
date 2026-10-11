"""Ordered teardown and current-launch completion proof for train-then-eval."""

import argparse
import json
import os
from pathlib import Path
import uuid


def completion_dir(run_dir, launch_id):
    # A fresh UUID prevents an earlier run's checkpoint from authorizing eval.
    if uuid.UUID(launch_id).hex != launch_id:
        raise ValueError("Invalid training launch ID")
    return Path(run_dir) / ".train_completion" / launch_id


def record_training_complete(trainer):
    launch_id = os.environ.get("GRAPH_TRAIN_LAUNCH_ID")
    if not launch_id:
        return
    if trainer.step != trainer.config.total_updates or trainer.pending is not None:
        raise RuntimeError("Training returned before completing all optimizer steps")
    path = completion_dir(trainer.config.output_dir, launch_id)
    path.mkdir(parents=True, exist_ok=True)
    final = path / f"rank_{trainer.rank}.json"
    temp = final.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {"launch_id": launch_id, "rank": trainer.rank, "world": trainer.world, "step": trainer.step}
        )
        + "\n"
    )
    os.replace(temp, final)


def validate_completion(run_dir, launch_id, steps, world):
    path = completion_dir(run_dir, launch_id)
    for rank in range(world):
        record = json.loads((path / f"rank_{rank}.json").read_text())
        if record != {"launch_id": launch_id, "rank": rank, "world": world, "step": steps}:
            raise ValueError(f"Rank {rank} did not complete this launch's requested steps")
    checkpoint = Path(run_dir) / f"checkpoint-{steps}"
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if manifest["step"] != steps or manifest["pending_window"]:
        raise ValueError("Final checkpoint does not contain a completed training horizon")
    if manifest["method_version"] != "graph_virtual_adam_atesd_v1":
        raise ValueError("Final checkpoint belongs to another method")
    json.loads((checkpoint / "adapter_config.json").read_text())
    if not (checkpoint / "training_state.pt").stat().st_size:
        raise ValueError("Empty final training state")
    from safetensors import safe_open

    with safe_open(checkpoint / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        if not tensors.keys():
            raise ValueError("Empty final adapter")


def shutdown_runtime(trainer, rank):
    import torch
    import torch.distributed as dist

    rollout = trainer.rollout_engine if trainer is not None else None
    native = rollout is not None and rollout.native_runtime
    try:
        if rollout is not None:
            print(f"[rank {rank}] shutdown: closing rollout engine", flush=True)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            rollout.close()
            trainer.rollout_engine = None
    finally:
        try:
            if native:
                print(f"[rank {rank}] shutdown: destroying vLLM groups", flush=True)
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
        finally:
            if dist.is_initialized():
                print(f"[rank {rank}] shutdown: destroying training process group", flush=True)
                dist.destroy_process_group()
    print(f"[rank {rank}] shutdown: complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--world", required=True, type=int)
    args = parser.parse_args()
    validate_completion(args.run_dir, args.launch_id, args.steps, args.world)
    print(f"Verified all {args.world} ranks completed {args.steps} steps and saved the final checkpoint.")


if __name__ == "__main__":
    main()
