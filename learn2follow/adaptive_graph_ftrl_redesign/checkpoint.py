"""Atomic redesign checkpoints, including pending reward and independent RNGs."""

from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from .rng import restore_rng, rng_state


def save_checkpoint(trainer):
    ranks = trainer.gather(
        {
            "rng": rng_state(),
            "mask_rng": trainer.mask_rng.bit_generator.state,
            "grounded_rng": trainer.grounded_rng.bit_generator.state,
        }
    )
    if trainer.rank == 0:
        root = Path(trainer.config.output_dir)
        final, temp = root / f"checkpoint-{trainer.step}", root / f".checkpoint-{trainer.step}.tmp"
        if final.exists() or temp.exists():
            raise FileExistsError(f"Checkpoint exists: {final} or {temp}")
        temp.mkdir(parents=True)
        trainer.model.save_pretrained(temp, safe_serialization=True)
        trainer.tokenizer.save_pretrained(temp)
        torch.save(
            {
                "config": asdict(trainer.config),
                "metadata": trainer.metadata,
                "identity": trainer.identity,
                "world": trainer.world,
                "step": trainer.step,
                "phase": trainer.phase,
                "window_index": trainer.window_index,
                "pending": trainer.pending,
                "main_ids": trainer.main_ids,
                "control_ids": trainer.control_ids,
                "main_sampler": trainer.main_sampler.state_dict(),
                "control_sampler": trainer.control_sampler.state_dict(),
                "ranks": ranks,
                "controller": trainer.controller.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
            },
            temp / "training_state.pt",
        )
        (temp / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "method_version": trainer.config.method_version,
                    "step": trainer.step,
                    "phase": trainer.phase,
                    "window_index": trainer.window_index,
                    "pending_window": trainer.pending is not None,
                    "full_opsd_warmup_steps": trainer.config.full_opsd_warmup_steps,
                    "answer_visibility": trainer.config.mask_teacher_include_final_answer,
                    "teacher_selection": trainer.config.teacher_selection,
                    "latent_weights": trainer.controller.weights,
                    "effective_weights": (
                        trainer.controller.weights
                        if trainer.phase == "adaptive" and trainer.config.teacher_selection == "hard"
                        else trainer.pending["weights_used"]
                        if trainer.pending
                        else ([1, 0] if trainer.phase == "full" else trainer.controller.weights)
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temp, final)
        if trainer.progress is not None:
            trainer.progress.write(f"Saved checkpoint: {final}")
        else:
            print(f"Saved checkpoint: {final}", flush=True)
    trainer.broadcast(True)


def restore_checkpoint(trainer, path):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(path)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    current = asdict(trainer.config)
    if state["config"].get("method_version") != current["method_version"]:
        raise ValueError("Resume method/reward/control definition differs; start a fresh redesign run")
    for key, value in current.items():
        if (
            key not in {"output_dir", "dataset", "verbose_runtime", "replay_cache_dir"}
            and state["config"].get(key, {"teacher_selection": "soft", "delta_scaled_clip": 1.0}.get(key))
            != value
        ):
            raise ValueError(f"Resume config differs: {key}")
    if (
        state["metadata"] != trainer.metadata
        or state["identity"] != trainer.identity
        or state["world"] != trainer.world
    ):
        raise ValueError("Resume data/parser/model/tokenizer/world differs")
    if state["main_ids"] != trainer.main_ids or state["control_ids"] != trainer.control_ids:
        raise ValueError("Resume question partition differs")
    result = set_peft_model_state_dict(trainer.model, load_file(str(path / "adapter_model.safetensors")))
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.controller.restore(state["controller"])
    trainer.main_sampler.restore(state["main_sampler"])
    trainer.control_sampler.restore(state["control_sampler"])
    trainer.step, trainer.window_index, trainer.pending = (
        state["step"],
        state["window_index"],
        state["pending"],
    )
    if trainer.phase != state["phase"]:
        raise ValueError("Resume phase differs")
    rank = state["ranks"][trainer.rank]
    trainer.mask_rng.bit_generator.state = rank["mask_rng"]
    trainer.grounded_rng.bit_generator.state = rank["grounded_rng"]
    if trainer.rollout_engine is not None:
        trainer.rollout_engine.synced_step = None
    logs = [(trainer.samples.path, "step", trainer.step)]
    if trainer.rank == 0:
        logs += [
            (trainer.metrics.path, "step", trainer.step),
            (trainer.windows.path, "end_step", trainer.step),
            (
                trainer.controls.path,
                "window_index",
                trainer.window_index if trainer.pending else trainer.window_index - 1,
            ),
        ]
    for log, key, limit in logs:
        if log.exists():
            temp = log.with_suffix(".resume.tmp")
            with log.open() as reader, temp.open("w") as writer:
                for line in reader:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row[key] <= limit:
                        writer.write(line)
            os.replace(temp, log)
    restore_rng(rank["rng"])
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.broadcast(True)
