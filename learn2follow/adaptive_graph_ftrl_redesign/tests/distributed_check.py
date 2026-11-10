"""Four Gloo ranks and mocked vLLM: full batch duel, phase transition, resume."""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import torch
import torch.distributed as dist

from adaptive_graph_ftrl_redesign.rollout import VLLMRollouts
from .rollout_fixture import FakeEngine, request

from adaptive_graph_ftrl_redesign.checkpoint import restore_checkpoint, save_checkpoint
from adaptive_graph_ftrl_redesign.lifecycle import (
    record_training_complete,
    shutdown_runtime,
    validate_completion,
)
from .fixtures import tiny_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-selection", choices=["soft", "hard"], default="soft")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo")
    trainer = None
    try:
        trainer = tiny_trainer(
            Path(args.output_dir),
            rank=rank,
            world=world,
            gradient_accumulation_steps=8,
            effective_batch_size=8 * world,
            control_pool_size=8,
            control_batch_size=8,
            min_valid_controls=4,
            grounded_score_examples_per_step=4,
            generation_backend="vllm",
            teacher_selection=args.teacher_selection,
            total_updates=4,
        )
        del trainer.generate_many

        class Engine(FakeEngine):
            def shutdown(self):
                self.closed = True

            def generate(self, prompts, sampling_params, lora_request, use_tqdm):
                assert not self.asleep
                assert not trainer.generating_control
                self.calls.append((prompts, sampling_params, lora_request))
                tokens = trainer.tokenizer.encode("x+y=3.") + [2]
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=tokens)]) for _ in prompts]

        trainer.rollout_engine = VLLMRollouts(trainer, (Engine, lambda **kw: SimpleNamespace(**kw), request))
        torch.manual_seed(42 + rank)
        first = trainer.run_update()
        assert first["counts"]["full_teacher_forward_calls"] == 32
        assert first["counts"].get("mask_teacher_forward_calls", 0) == 0
        trainer.run_update()
        adaptive = trainer.run_update()
        assert trainer.pending["phase"] == "adaptive" and trainer.pending["steps_done"] == 1
        if args.teacher_selection == "hard":
            assert adaptive["counts"]["pure_expert_gradient_calls"] == 64
            assert adaptive["expert_feedback"]["ftrl_updated"]
            assert adaptive["weights_used"] in ([1, 0], [0, 1])
            assert "replay_student_forward_calls" not in adaptive["counts"]
        save_checkpoint(trainer)
        expected = trainer.run_update()
        assert expected["window_feedback"]["duel_valid"] and expected["window_feedback"]["actor_valid"]
        assert expected["counts"]["main_student_forward_calls"] == 32
        assert expected["counts"]["pure_expert_gradient_calls"] == 64
        assert expected["counts"]["control_forward_calls"] == 24
        assert expected["counts"]["grounded_teacher_forward_calls"] == 4
        assert expected["counts"]["replay_student_forward_calls"] == 32
        assert expected["counts"]["vllm_generation_batches"] == 4
        assert all(len(call[0]) == 8 for call in trainer.rollout_engine.engine.calls)
        parameters = torch.cat([p.detach().flatten() for p in trainer.params.values()])
        gathered = trainer.gather(parameters.tolist())
        assert all(p == gathered[0] for p in gathered)
        weights = trainer.gather(trainer.controller.weights)
        assert all(w == weights[0] for w in weights)
        restore_checkpoint(trainer, Path(args.output_dir) / "checkpoint-3")
        actual = trainer.run_update()
        assert actual["loss"] == expected["loss"] and actual["window_feedback"] == expected["window_feedback"]
        assert actual.get("expert_feedback") == expected.get("expert_feedback")
        assert torch.equal(parameters, torch.cat([p.detach().flatten() for p in trainer.params.values()]))
        save_checkpoint(trainer)
        os.environ["GRAPH_TRAIN_LAUNCH_ID"] = trainer.broadcast(uuid.uuid4().hex if rank == 0 else None)
        record_training_complete(trainer)
        dist.barrier()
        if rank == 0:
            validate_completion(args.output_dir, os.environ["GRAPH_TRAIN_LAUNCH_ID"], 4, world)
        engine = trainer.rollout_engine.engine
        shutdown_runtime(trainer, rank)
        assert engine.closed and trainer.rollout_engine is None
        if rank == 0:
            print(
                "PASSED: four ranks, vLLM mock, accu=8 batch=32, FULL warmup, full-batch exact AdamW duel, reference scoring, frozen-target replay, synchronized weights, exact mid-window resume, completion proof and ordered shutdown.",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
