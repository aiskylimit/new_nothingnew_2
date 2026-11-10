"""Train the self-contained adaptive graph redesign."""

from dataclasses import asdict
import faulthandler
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist

from .data import load_examples
from .benchmark_partition import benchmark_question_ids
from .model import load_runtime

from .config import load_config
from .cli import parser
from .lifecycle import record_training_complete, shutdown_runtime
from .trainer import Trainer


def main():
    faulthandler.enable(all_threads=True)
    args = parser().parse_args()
    cfg = load_config(
        args.config, {k: v for k, v in vars(args).items() if k not in {"config", "resume", "validate_only"}}
    )
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    cfg.validate(world)
    examples, metadata = load_examples(cfg.dataset, cfg.data_limit)
    benchmark = benchmark_question_ids(cfg.benchmark_tasks)
    excluded = set(benchmark["benchmark_question_ids"])
    metadata["benchmark_excluded"] = sum(ex.qid in excluded for ex in examples)
    metadata["benchmark_question_ids"] = sorted(excluded)
    examples = [ex for ex in examples if ex.qid not in excluded]
    if len(examples) <= cfg.control_pool_size:
        raise ValueError("Control pool must leave main examples")
    output = Path(cfg.output_dir)
    if rank == 0:
        print(
            f"Graph FTRL Redesign | model={cfg.model} variant={cfg.variant} answer={cfg.mask_teacher_include_final_answer}\n"
            f"steps={cfg.total_updates} FULL_warmup={cfg.full_opsd_warmup_steps} H={cfg.window_steps} batch={cfg.effective_batch_size} accu={cfg.gradient_accumulation_steps}\n"
            f"train_seed={cfg.seed} eval_seed={cfg.eval_seed} train_temp={cfg.generation_temperature} eval_temp={cfg.eval_temperature} backend={cfg.generation_backend}\n"
            f"teacher_thinking={cfg.teacher_thinking} student_thinking={cfg.student_thinking} controls={cfg.control_batch_size}/{cfg.control_pool_size} min_valid={cfg.min_valid_controls}\n"
            f"gamma={cfg.gamma_discount} lambda_gt={cfg.lambda_gt} grounded_subset={cfg.grounded_score_examples_per_step} reference_cap={cfg.grounded_reference_max_tokens} eta={cfg.ftrl_eta if cfg.teacher_selection == 'soft' else 'unused_hard_leader'}\n"
            f"teacher_selection={cfg.teacher_selection} expert_refresh_steps={1 if cfg.teacher_selection == 'hard' else cfg.window_steps}\n"
            f"ftrl_rho={cfg.ftrl_rho} delta_rms_decay={cfg.delta_rms_decay} delta_rms_floor={cfg.delta_rms_floor} delta_scaled_clip={cfg.delta_scaled_clip} ftrl_min_weight={cfg.ftrl_min_weight if cfg.teacher_selection == 'soft' else 'unused_hard_leader'}\n"
            f"actor_warmup_windows={cfg.warmup_windows} controller_lr={cfg.controller_lr} exposure_temperature={cfg.exposure_temperature} learning_rate={cfg.learning_rate}\n"
            f"eligible={len(examples)} excluded={metadata['excluded']} output={output}",
            flush=True,
        )
    if args.validate_only:
        return
    if not args.resume and any(output.glob("checkpoint-*")):
        raise FileExistsError("Use --resume or a fresh run directory")
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    trainer = None
    try:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        model, tokenizer = load_runtime(cfg, device)
        torch.manual_seed(cfg.seed + rank)
        random.seed(cfg.seed + rank)
        np.random.seed(cfg.seed + rank)
        trainer = Trainer(model, tokenizer, examples, cfg, metadata, rank, world)
        if args.resume:
            from .checkpoint import restore_checkpoint

            restore_checkpoint(trainer, args.resume)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "config": asdict(cfg),
                        "metadata": {k: v for k, v in metadata.items() if k != "invalid_rows"},
                        "world": world,
                        "main_ids": trainer.main_ids,
                        "control_ids": trainer.control_ids,
                        "branching_rate": float(np.mean([trainer.branching[q] for q in trainer.main_ids])),
                    },
                    indent=2,
                )
                + "\n"
            )
            with (output / "invalid_rows.jsonl").open("w") as handle:
                for row in metadata["invalid_rows"]:
                    handle.write(json.dumps(row) + "\n")
        trainer.train()
        record_training_complete(trainer)
        if dist.is_initialized():
            dist.barrier()
    finally:
        training_error = sys.exc_info()[1]
        try:
            shutdown_runtime(trainer, rank)
        except Exception:
            if training_error is None:
                raise
            traceback.print_exc()


if __name__ == "__main__":
    main()
