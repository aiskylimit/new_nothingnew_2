"""Offline tiny Qwen3/LoRA smoke, with no pretrained model download."""

import argparse
from pathlib import Path
import tempfile

import torch

from .checkpoint import restore_checkpoint, save_checkpoint
from .tests.fixtures import tiny_trainer


def smoke(path, teacher_selection="soft"):
    trainer = tiny_trainer(path / "original", teacher_selection=teacher_selection)
    for _ in range(3):
        trainer.run_update()
    save_checkpoint(trainer)
    expected = [trainer.run_update() for _ in range(7)]
    final = {n: p.detach().clone() for n, p in trainer.params.items()}
    resumed = tiny_trainer(path / "resumed", teacher_selection=teacher_selection)
    restore_checkpoint(resumed, path / "original" / "checkpoint-3")
    actual = [resumed.run_update() for _ in range(7)]
    assert [r["loss"] for r in actual] == [r["loss"] for r in expected]
    assert all(torch.equal(p, final[n]) for n, p in resumed.params.items())
    assert trainer.controller.actor_valid_windows == 4
    assert trainer.controller.duel_valid_windows == (8 if teacher_selection == "hard" else 4)
    assert trainer.controller.version == 1
    print(
        f"PASSED ({teacher_selection}): answer split -> graph masks -> FULL warmup -> student update -> grounded reference -> exact virtual duel -> expert update -> actor warmup/update -> bit-exact resume."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--teacher-selection", choices=["soft", "hard"], default="soft")
    args = parser.parse_args()
    if args.output_dir:
        smoke(Path(args.output_dir), args.teacher_selection)
    else:
        with tempfile.TemporaryDirectory(prefix="redesign-smoke-") as directory:
            smoke(Path(directory), args.teacher_selection)


if __name__ == "__main__":
    main()
