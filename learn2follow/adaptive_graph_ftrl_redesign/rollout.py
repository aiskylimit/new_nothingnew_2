"""Batched, current-snapshot vLLM rollouts without merging the training base."""

from pathlib import Path
import gc
import os
import time

import torch

from .rng import rng_state, restore_rng


class VLLMRollouts:
    def __init__(self, trainer, factories=None):
        self.trainer = trainer
        cfg = trainer.config
        self.native_runtime = factories is None
        if factories is None:
            if trainer.device.type != "cuda":
                raise RuntimeError("generation_backend=vllm requires CUDA")
            if any(key not in os.environ for key in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
                raise RuntimeError("Launch the vLLM training backend with torchrun")
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            if not getattr(cfg, "verbose_runtime", True):
                os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
            try:
                from vllm import LLM, SamplingParams
                from vllm.lora.request import LoRARequest
            except ImportError as error:
                raise RuntimeError("Install vllm in the active training environment") from error
            factories = LLM, SamplingParams, LoRARequest
        llm, self.sampling, self.request = factories
        state = rng_state()
        try:
            self.engine = llm(
                model=cfg.model,
                dtype=cfg.dtype,
                tensor_parallel_size=1,
                distributed_executor_backend="external_launcher",
                gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                max_model_len=cfg.max_length,
                max_num_seqs=max(cfg.gradient_accumulation_steps, cfg.probe_batch_size),
                enable_lora=True,
                max_lora_rank=cfg.lora_rank,
                max_loras=1,
                max_cpu_loras=1,
                enable_sleep_mode=True,
                seed=cfg.seed + trainer.rank,
                enable_prefix_caching=False,
                generation_config="vllm",
            )
        finally:
            restore_rng(state)
        self.adapter = None
        self.synced_step = None
        self.sleeping = False
        self.path = Path(cfg.output_dir).resolve() / ".rollout_adapters" / f"rank_{trainer.rank}"

    def generate(self, prompts, budget):
        if self.engine is None:
            raise RuntimeError("Rollout engine is closed")
        trainer, cfg = self.trainer, self.trainer.config
        seeds = torch.randint(0, 2**31 - 1, (len(prompts),)).tolist()
        state = rng_state()
        started = time.monotonic()
        try:
            if trainer.device.type == "cuda":
                torch.cuda.empty_cache()
            if self.sleeping:
                self.engine.wake_up()
                self.sleeping = False
            if self.synced_step != trainer.step:
                if self.adapter is not None:
                    self.engine.llm_engine.remove_lora(self.adapter.lora_int_id)
                # Export LoRA only: merge/unmerge would round the frozen bf16 base.
                self.path.mkdir(parents=True, exist_ok=True)
                trainer.model.save_pretrained(self.path, safe_serialization=True)
                self.adapter = self.request(f"round_{trainer.step + 1}", trainer.step + 1, str(self.path))
                self.synced_step = trainer.step
                trainer.count("rollout_adapter_sync_calls")
            trainer.timings["rollout_wake_sync"] = (
                trainer.timings.get("rollout_wake_sync", 0.0) + time.monotonic() - started
            )
            # Per-request seeds use checkpointed torch RNG, not hidden engine RNG.
            params = [
                self.sampling(
                    n=1,
                    temperature=cfg.generation_temperature,
                    top_p=cfg.top_p,
                    top_k=-1,
                    min_p=0.0,
                    presence_penalty=0.0,
                    max_tokens=budget,
                    seed=seed,
                    stop_token_ids=trainer.model.generation_config.eos_token_id
                    if isinstance(trainer.model.generation_config.eos_token_id, list)
                    else [trainer.model.generation_config.eos_token_id]
                    if trainer.model.generation_config.eos_token_id is not None
                    else [],
                )
                for seed in seeds
            ]
            outputs = self.engine.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts],
                sampling_params=params,
                lora_request=self.adapter,
                use_tqdm=False,
            )
            trainer.count("vllm_generation_batches")
            if len(outputs) != len(prompts) or any(len(out.outputs) != 1 for out in outputs):
                raise RuntimeError("vLLM must return exactly one completion per question")
            return [list(out.outputs[0].token_ids) for out in outputs]
        finally:
            started = time.monotonic()
            # Level 1 preserves base and LoRA weights; level 2 would discard them.
            try:
                self.engine.sleep(level=1)
                self.sleeping = True
                trainer.timings["rollout_sleep"] = (
                    trainer.timings.get("rollout_sleep", 0.0) + time.monotonic() - started
                )
            finally:
                restore_rng(state)

    @staticmethod
    def _shutdown_engine(engine):
        llm = engine.llm_engine
        core = getattr(llm, "engine_core", None)
        if core is not None:
            core.shutdown()
        elif callable(getattr(llm, "shutdown", None)):
            llm.shutdown()
        else:
            llm.model_executor.shutdown()

    def close(self):
        """Release the engine while CUDA, allocators and communicators still live."""
        if self.engine is None:
            return
        engine, self.engine = self.engine, None
        try:
            self._shutdown_engine(engine)
        finally:
            self.adapter, self.synced_step, self.trainer = None, None, None
            del engine
            gc.collect()
