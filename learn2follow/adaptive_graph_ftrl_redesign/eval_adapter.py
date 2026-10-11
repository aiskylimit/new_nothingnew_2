"""Adapters to existing repository entry points."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def eval_command(
    checkpoint_dir: str,
    dataset: str,
    gpus: str = "0,1,2,3",
    seed: int = 42,
    max_new_tokens: int = 38912,
    temperature: float = 1.0,
    top_p: float = 0.95,
    output_file: str | None = None,
    model: str = "Qwen/Qwen3-4B",
    top_k: int = -1,
    samples: int = 12,
    gpu_memory_utilization: float = 0.9,
    launcher: str = "python",
    dataset_dir: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    dataset_args = ["--dataset_dir", dataset_dir] if dataset_dir else []
    if output_file is None:
        output_file = str(
            Path("full_masked_graph_opsd/outputs/evaluation")
            / Path(checkpoint_dir).parent.name
            / f"{Path(checkpoint_dir).name}_{dataset}.json"
        )
    if launcher == "se-opsd":
        return [
            "bash",
            "graph_based_scripts/run_graph_based_eval_opsd_4gpu.sh",
            "--gpu_memory_utilization",
            str(gpu_memory_utilization),
            "--enable_thinking",
            "--output_file",
            output_file,
        ] + dataset_args, {
            "CUDA_VISIBLE_DEVICES": gpus,
            "MODEL_PATH": model,
            "CHECKPOINT_DIR": checkpoint_dir,
            "DATASET": dataset,
            "MAX_NEW_TOKENS": str(max_new_tokens),
            "TEMPERATURE": str(temperature),
            "TOP_P": str(top_p),
            "TOP_K": str(top_k),
            "MIN_P": "0.0",
            "PRESENCE_PENALTY": "0.0",
            "VAL_N": str(samples),
            "TENSOR_PARALLEL_SIZE": str(len(gpus.split(","))),
            "EVAL_SEED": str(seed),
        }
    if launcher != "python":
        raise ValueError(f"Unsupported evaluation launcher: {launcher}")
    return [
        sys.executable,
        "eval/evaluate_math.py",
        "--base_model",
        model,
        "--checkpoint_dir",
        checkpoint_dir,
        "--dataset",
        dataset,
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--min_p",
        "0.0",
        "--presence_penalty",
        "0.0",
        "--val_n",
        str(samples),
        "--tensor_parallel_size",
        str(len(gpus.split(","))),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--seed",
        str(seed),
        "--enable_thinking",
        "--output_file",
        output_file,
    ] + dataset_args, {"CUDA_VISIBLE_DEVICES": gpus}


def run_eval(checkpoint_dir: str, dataset: str, output_log: str | None = None, **kwargs) -> int:
    cmd, env_updates = eval_command(checkpoint_dir, dataset, **kwargs)
    import os

    env = os.environ.copy()
    env.update(env_updates)
    if output_log:
        Path(output_log).parent.mkdir(parents=True, exist_ok=True)
        with Path(output_log).open("w", encoding="utf-8") as handle:
            return subprocess.run(
                cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
            ).returncode
    return subprocess.run(cmd, env=env, check=False).returncode


def shell_eval_line(checkpoint_dir: str, dataset: str, **kwargs) -> str:
    cmd, env = eval_command(checkpoint_dir, dataset, **kwargs)
    assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"{assigns} {' '.join(shlex.quote(part) for part in cmd)}"
