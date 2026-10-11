"""Shared mechanics for the redesign trainer, without legacy method imports."""

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .masks import candidate_pool
from .metrics import MetricsWriter
from .prompts import student_user_message, teacher_user_message
from .sampler import Sampler


FINAL_ANSWER_HEADING = re.compile(r"[\s#*]*Final\s+answer\s*:\s*(?:\s|[*$]|\\\[)*\Z", re.I)


class Runtime:
    def __init__(self, model, tokenizer, examples, config, metadata, rank=0, world=1):
        self.model, self.tokenizer, self.config = model, tokenizer, config.validate(world)
        self.rank, self.world, self.step = rank, world, 0
        self.device = next(model.parameters()).device
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not self.params or any(
            "lora_" not in n or p.dtype != torch.float32 for n, p in self.params.items()
        ):
            raise ValueError("Student optimizer requires FP32 trainable LoRA only")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        if hasattr(model.config, "attention_dropout"):
            model.config.attention_dropout = 0.0
        self.examples = {ex.qid: ex for ex in examples}
        if len(self.examples) != len(examples):
            raise ValueError("Duplicate question IDs")
        order = sorted(self.examples, key=lambda q: hashlib.sha256(f"{config.seed}:{q}".encode()).digest())
        if not 0 < config.control_pool_size < len(order):
            raise ValueError("Control pool must leave main examples")
        self.control_ids = order[: config.control_pool_size]
        self.main_ids = order[config.control_pool_size :]
        self.main_sampler = Sampler(self.main_ids, config.seed + 1)
        self.control_sampler = Sampler(self.control_ids, config.seed + 2)
        self.mask_rng = np.random.default_rng(config.seed + rank + 1000)
        self.pools = {}
        self.controller = self.controller_class(config)
        if config.variant == "full":
            self.controller.weights = [1.0, 0.0]
        self.optimizer = torch.optim.AdamW(
            list(self.params.values()),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: max(0.0, 1 - s / config.total_updates)
        )
        self.window_index, self.pending = 0, None
        self.generating_control = False
        self.progress = None
        self.rollout_engine = None
        self.metadata, self.counts, self.timings = metadata, {}, {}
        self.metrics = MetricsWriter(Path(config.output_dir) / "metrics.jsonl")
        self.windows = MetricsWriter(Path(config.output_dir) / "windows.jsonl")
        self.controls = MetricsWriter(Path(config.output_dir) / "controls.jsonl")
        self.samples = MetricsWriter(Path(config.output_dir) / f"samples_rank_{rank}.jsonl")
        self.identity = hashlib.sha256(
            json.dumps(
                {
                    "model": model.config.to_dict(),
                    "tokenizer": tokenizer.get_vocab(),
                    "special": tokenizer.all_special_ids,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        base = model.get_base_model()
        if "logits_to_keep" not in inspect.signature(base.forward).parameters:
            raise RuntimeError("Model must support logits_to_keep")
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def prompt(self, example, reference=None):
        if reference is not None:
            text, thinking = teacher_user_message(example.problem, reference), True
        else:
            text, thinking = student_user_message(example.problem), False
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def pool(self, example):
        if example.qid not in self.pools:
            self.pools[example.qid] = candidate_pool(
                example,
                self.tokenizer,
                self.config.variant,
                self.config.max_pair_candidates,
                self.config.seed,
            )
        return self.pools[example.qid]

    def answer_loss(self, record):
        prefix, labels = record["prefix"], record["labels"]
        ids = torch.tensor([prefix + labels], device=self.device)
        output = self.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=len(labels) + 1,
        )
        logits = output.logits[0, -len(labels) - 1 : -1].clone()
        del output
        losses = []
        for chunk in self.chunks(len(labels)):
            losses.append(
                F.cross_entropy(
                    logits[chunk].float() / self.config.answer_temperature,
                    torch.tensor(labels[chunk], device=self.device),
                    reduction="sum",
                )
                / len(labels)
            )
        return torch.stack(losses).sum()

    def control_value(self, records, gradients=False):
        valid = [r for r in records if r["valid"]]
        if not valid:
            return None, None
        self.model.eval()
        grads = {n: torch.zeros_like(p) for n, p in self.params.items()} if gradients else None
        total = torch.zeros((), device=self.device)
        with self.timed("control_gradient" if gradients else "control_score"):
            for record in valid[self.rank :: self.world]:
                with torch.enable_grad() if gradients else torch.no_grad():
                    loss = self.answer_loss(record)
                    total += loss.detach() / len(valid)
                    if gradients:
                        values = torch.autograd.grad(
                            loss / len(valid), tuple(self.params.values()), allow_unused=True
                        )
                        for (name, _), value in zip(self.params.items(), values):
                            if value is not None:
                                grads[name].add_(value.detach())
                self.count("control_forward_calls")
                self.count("control_scored_tokens", len(record["prefix"]) + len(record["labels"]))
        if self.world > 1:
            dist.all_reduce(total)
            if gradients:
                for value in grads.values():
                    dist.all_reduce(value)
        return float(total), grads

    def gather(self, value):
        if self.world == 1:
            return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value)
        return result

    def broadcast(self, value):
        if self.world == 1:
            return value
        result = [value if self.rank == 0 else None]
        dist.broadcast_object_list(result, src=0)
        return result[0]

    def count(self, key, amount=1):
        self.counts[key] = self.counts.get(key, 0) + amount

    @contextmanager
    def timed(self, key):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.monotonic()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.timings[key] = self.timings.get(key, 0.0) + time.monotonic() - start

    def generate(self, ex, probe=False):
        if self.config.generation_backend == "vllm":
            return self.generate_many([ex], probe=probe)[0]
        self.model.eval()
        prompt = self.prompt(ex)
        budget = self.config.probe_max_new_tokens if probe else self.config.main_max_new_tokens
        if len(prompt) + budget > self.config.max_length:
            raise ValueError(f"{ex.qid}: question + generation exceeds max_length; no silent truncation")
        ids = torch.tensor([prompt], device=self.device)
        prefix = "probe" if probe else "main"
        started = time.monotonic()
        print(
            f"[rank {self.rank}] {prefix} generation start step={self.step + 1} "
            f"qid={ex.qid[:12]} prompt_tokens={len(prompt)} budget={budget}",
            flush=True,
        )
        with self.timed(prefix + "_generation"), torch.no_grad():
            output = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=budget,
                do_sample=True,
                temperature=self.config.generation_temperature,
                top_p=self.config.top_p,
                top_k=0,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        tokens = output[0, len(prompt) :].detach().cpu().tolist()
        if not tokens:
            raise RuntimeError(f"Empty rollout {ex.qid}")
        self.count(prefix + "_generation_calls")
        self.count(prefix + "_generated_tokens", len(tokens))
        self.count(prefix + "_generation_prompt_tokens", len(prompt))
        self.count(prefix + "_generation_model_forward_calls", len(tokens))
        print(
            f"[rank {self.rank}] {prefix} generation done step={self.step + 1} qid={ex.qid[:12]} "
            f"tokens={len(tokens)} seconds={time.monotonic() - started:.2f}",
            flush=True,
        )
        return tokens

    def generate_many(self, examples, probe=False):
        if not examples:
            return []
        if self.config.generation_backend == "transformers":
            return [self.generate(ex, probe=probe) for ex in examples]
        from .rollout import VLLMRollouts

        self.model.eval()
        budget = self.config.probe_max_new_tokens if probe else self.config.main_max_new_tokens
        prompts = [self.prompt(ex) for ex in examples]
        for ex, prompt in zip(examples, prompts):
            if len(prompt) + budget > self.config.max_length:
                raise ValueError(f"{ex.qid}: question + generation exceeds max_length")
        prefix = "probe" if probe else "main"
        if getattr(self.config, "verbose_runtime", True):
            print(
                f"[rank {self.rank}] {prefix} vLLM batch start step={self.step + 1} "
                f"prompts={len(prompts)} budget={budget} snapshot={self.step + 1}",
                flush=True,
            )
        with self.timed(prefix + "_generation"):
            if self.rollout_engine is None:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                self.rollout_engine = VLLMRollouts(self)
            outputs = self.rollout_engine.generate(prompts, budget)
        for ex, prompt, tokens in zip(examples, prompts, outputs):
            if not tokens or len(tokens) > budget:
                raise RuntimeError(f"Invalid rollout length {ex.qid}: {len(tokens)}")
            self.count(prefix + "_generation_calls")
            self.count(prefix + "_generated_tokens", len(tokens))
            self.count(prefix + "_generation_prompt_tokens", len(prompt))
            if getattr(self.config, "verbose_runtime", True):
                print(
                    f"[rank {self.rank}] {prefix} generation done step={self.step + 1} "
                    f"qid={ex.qid[:12]} tokens={len(tokens)} backend=vllm",
                    flush=True,
                )
        return outputs

    def logits_kwargs(self, ex, tokens, reference=None):
        prompt = self.prompt(ex, reference)
        if len(prompt) + len(tokens) > self.config.max_length:
            raise ValueError(f"{ex.qid}: reference + rollout exceeds max_length; increase it explicitly")
        ids = torch.tensor([prompt + tokens], device=self.device)
        return dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            logits_to_keep=len(tokens) + 1,
        )

    def logits(self, ex, tokens, reference=None):
        output = self.model(**self.logits_kwargs(ex, tokens, reference))
        return output.logits[0, -len(tokens) - 1 : -1].clone()

    def score(self, ex, tokens, reference, prefix):
        self.model.eval()
        with self.timed(prefix + "_score"), torch.no_grad():
            value = self.logits(ex, tokens, reference).detach().cpu()
        self.count(prefix + "_forward_calls")
        self.count(prefix + "_scored_tokens", len(self.prompt(ex, reference)) + len(tokens))
        return value

    def chunks(self, length):
        for start in range(0, length, self.config.chunk_size):
            yield slice(start, min(length, start + self.config.chunk_size))

    def probabilities(self, logits):
        return F.softmax(logits.to(self.device).float() / self.config.distillation_temperature, -1)

    def main_backward(self, ex, tokens, targets, local_batch):
        self.model.train()
        with self.timed("main_forward_backward"):
            logits = self.logits(ex, tokens)
            losses = []
            for chunk, target in zip(self.chunks(len(tokens)), targets):

                def chunk_loss(z, q):
                    logp = F.log_softmax(z.float() / self.config.distillation_temperature, -1)
                    q = q.to(z.device)
                    return (torch.xlogy(q, q) - q * logp).sum() / len(tokens)

                losses.append(checkpoint(chunk_loss, logits[chunk], target, use_reentrant=False))
            loss = torch.stack(losses).sum()
            value = float(loss.detach())
            (loss / local_batch).backward()
        self.count("main_student_forward_calls")
        self.count("main_student_scored_tokens", len(self.prompt(ex)) + len(tokens))
        return value
