"""Window-fixed graph exposure with soft aggregation or per-step hard leaders."""

import math
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from .data import FORMATTING, _boxed_span, render_mask
from .runtime import FINAL_ANSWER_HEADING, Runtime

from .controller import Controller, exposure_reward
from .duel import virtual_duel
from .replay import ReplayCache


def canonical_answer(gold):
    box = _boxed_span(gold)
    if "\\boxed" in gold and box is None:
        raise ValueError("Malformed boxed gold answer")
    if (
        box is not None
        and FORMATTING.fullmatch(FINAL_ANSWER_HEADING.sub("", gold[: box[0]]))
        and FORMATTING.fullmatch(gold[box[2] :])
    ):
        return gold[box[1] + 1 : box[2] - 1].strip()
    return re.sub(r"^\s*(?:#+\s*)?(?:\*\*)?Final\s+answer\s*:\s*(?:\*\*)?\s*", "", gold, flags=re.I).strip()


class Trainer(Runtime):
    controller_class = Controller

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Soft latent weights stay uniform even for FULL-only training.
        if self.config.teacher_selection == "soft":
            self.controller.weights = [0.5, 0.5]
        self.grounded_rng = np.random.default_rng(self.config.seed + 3000)
        self.reference_labels = {}
        self.branching = {}
        for ex in self.examples.values():
            children = {v: set() for v in ex.node_ids}
            for child, parents in ex.parents_by_child.items():
                for parent in parents:
                    children[parent].add(child)
            self.branching[ex.qid] = any(len(c) > 1 for c in children.values()) or any(
                len(p) > 1 for p in ex.parents_by_child.values()
            )

    @property
    def phase(self):
        return (
            "full"
            if self.config.variant == "full" or self.step < self.config.full_opsd_warmup_steps
            else "adaptive"
        )

    def control_inputs(self, example):
        cfg = self.config
        text = f"Problem: {example.problem}\n\n{cfg.direct_instruction}"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        record = {
            "qid": example.qid,
            "valid": False,
            "template": cfg.direct_instruction,
            "cue": cfg.direct_answer_cue,
        }
        try:
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False) + self.tokenizer.encode(
                cfg.direct_answer_cue, add_special_tokens=False
            )
            gold = canonical_answer(example.gold_answer)
            labels = self.tokenizer.encode(gold, add_special_tokens=False)
        except (ValueError, TypeError) as error:
            return {**record, "reason": "gold_tokenization_error", "error": str(error)}
        if not prefix or not labels:
            return {**record, "reason": "empty_gold_label_or_template"}
        if len(prefix) + len(labels) > cfg.max_length:
            return {**record, "reason": "direct_context_overflow"}
        return {
            **record,
            "valid": True,
            "reason": "",
            "prefix": prefix,
            "labels": labels,
            "gold_content": gold,
        }

    def pool(self, example):
        pool = super().pool(example)
        if any(cost <= 0 for cost in pool.token_lengths.values()):
            raise ValueError(f"{example.qid}: graph_ineligible: zero-token reasoning node")
        return pool

    def generate(self, example, probe=False):
        if probe:
            raise RuntimeError("Redesign never generates control reasoning")
        if self.config.verbose_runtime:
            return super().generate(example)
        self.model.eval()
        prompt = self.prompt(example)
        budget = self.config.main_max_new_tokens
        if len(prompt) + budget > self.config.max_length:
            raise ValueError(f"{example.qid}: question + generation exceeds max_length")
        ids = torch.tensor([prompt], device=self.device)
        with self.timed("main_generation"), torch.no_grad():
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
            raise RuntimeError(f"Empty rollout: {example.qid}")
        self.count("main_generation_calls")
        self.count("main_generated_tokens", len(tokens))
        self.count("main_generation_prompt_tokens", len(prompt))
        return tokens

    def begin_window(self):
        cfg, phase = self.config, self.phase
        length = min(cfg.window_steps, cfg.total_updates - self.step)
        if phase == "full":
            if cfg.variant != "full":
                length = min(length, cfg.full_opsd_warmup_steps - self.step)
            action = {
                "state": [0.0] * 4,
                "alpha": None,
                "weights_used": [1.0, 0.0],
                "policy_version": self.controller.version,
                "beta_a": 2.0,
                "beta_b": 2.0,
            }
            records = []
        else:
            qids = self.control_sampler.take(cfg.control_batch_size)
            records = [self.control_inputs(self.examples[q]) for q in qids]
            action = self.broadcast(self.controller.begin(self.window_index) if self.rank == 0 else None)
        self.pending = {
            **action,
            "phase": phase,
            "window_index": self.window_index,
            "start_step": self.step,
            "steps_done": 0,
            "length": length,
            "control": records,
            "loss_observations": [],
            "grounded_observations": [],
            "sample_stats": [],
            "window_counts": {},
            "window_timings": {},
            "wall_seconds": 0.0,
            "latent_weights_used": list(self.controller.weights),
        }
        if cfg.teacher_selection == "hard":
            self.pending.update(step_weights=[], expert_feedback=[])
        if self.rank == 0 and phase == "adaptive":
            self.controls.write(
                {
                    "window_index": self.window_index,
                    "start_step": self.step,
                    "control": records,
                    "alpha": action["alpha"],
                    "state_used": action["state"],
                    "weights_used": action["weights_used"],
                }
            )

    def grounded_labels(self, example):
        if example.qid not in self.reference_labels:
            cfg = self.config
            full = self.tokenizer.encode(example.reference_solution, add_special_tokens=False)
            # Bound by every unmasked rendering, independent of selected action and answer flag.
            longest = max(
                len(self.prompt(example, reference))
                for reference in (
                    example.reference_solution,
                    render_mask(example, frozenset(), True),
                    render_mask(example, frozenset(), False),
                )
            )
            cap = min(cfg.grounded_reference_max_tokens, cfg.max_length - longest)
            labels = full
            if len(full) > cap:
                answer = self.tokenizer.encode(example.final_answer_block, add_special_tokens=False)
                if cap <= len(answer):
                    labels = []
                else:
                    reasoning = self.tokenizer.encode(example.reasoning_source, add_special_tokens=False)
                    labels = reasoning[: cap - len(answer)] + answer
            self.reference_labels[example.qid] = {
                "labels": labels,
                "reference_total_tokens": len(full),
                "reference_scored_tokens": len(labels),
                "reference_fraction": len(labels) / max(len(full), 1),
                "reference_truncated": len(labels) < len(full),
                "reference_span": cfg.grounded_reference_span,
                "reference_cap": cap,
            }
        return self.reference_labels[example.qid]

    def grounded_score(self, example, reference):
        span = self.grounded_labels(example)
        labels = span["labels"]
        if not labels:
            return None, {
                **{k: v for k, v in span.items() if k != "labels"},
                "grounded_valid": False,
                "grounded_invalid_reason": "reference_answer_does_not_fit",
            }
        self.model.eval()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        with self.timed("grounded_teacher_score"), torch.no_grad():
            logits = self.logits(example, labels, reference)
            for chunk in self.chunks(len(labels)):
                logq = F.log_softmax(logits[chunk].float() / self.config.distillation_temperature, -1)
                total += logq.gather(
                    -1, torch.tensor(labels[chunk], device=self.device).unsqueeze(-1)
                ).sum() / len(labels)
        self.count("grounded_teacher_forward_calls")
        self.count("grounded_teacher_label_tokens", len(labels))
        self.count("grounded_teacher_scored_tokens", len(self.prompt(example, reference)) + len(labels))
        valid = bool(torch.isfinite(total))
        return float(total), {
            **{k: v for k, v in span.items() if k != "labels"},
            "grounded_valid": valid,
            "grounded_invalid_reason": "" if valid else "nonfinite_reference_score",
        }

    def expert_gradients(self, example, tokens, full, mask, targets):
        self.model.train()
        with self.timed("pure_expert_gradients"):
            logits = self.logits(example, tokens)
            with torch.no_grad():
                value = torch.zeros((), device=self.device, dtype=torch.float32)
                for chunk, q in zip(self.chunks(len(tokens)), targets):
                    q = q.to(self.device)
                    logp = F.log_softmax(logits[chunk].float() / self.config.distillation_temperature, -1)
                    value += (torch.xlogy(q, q) - q * logp).sum() / len(tokens)
                mixture_loss = float(value)
            results = []
            for index, teacher in enumerate((full, mask)):
                pieces = []
                for chunk in self.chunks(len(tokens)):

                    def loss_chunk(z, teacher_logits):
                        q = self.probabilities(teacher_logits).detach()
                        logp = F.log_softmax(z.float() / self.config.distillation_temperature, -1)
                        return (torch.xlogy(q, q) - q * logp).sum() / len(tokens)

                    pieces.append(checkpoint(loss_chunk, logits[chunk], teacher[chunk], use_reentrant=False))
                loss = torch.stack(pieces).sum()
                grads = torch.autograd.grad(
                    loss, tuple(self.params.values()), retain_graph=index == 0, allow_unused=True
                )
                results.append(
                    {
                        n: torch.zeros_like(p) if g is None else g.detach()
                        for (n, p), g in zip(self.params.items(), grads)
                    }
                )
                self.count("pure_expert_gradient_calls")
            del logits, pieces, grads, loss
        self.count("main_student_forward_calls")
        self.count("main_student_scored_tokens", len(self.prompt(example)) + len(tokens))
        return mixture_loss, results

    def run_update(self):
        cfg, started = self.config, time.monotonic()
        self.counts, self.timings = {"teacher_generation_calls": 0, "control_generation_calls": 0}, {}
        self.optimizer.zero_grad(set_to_none=True)
        if self.pending is None:
            self.begin_window()
        pending = self.pending
        active = pending["phase"] == "adaptive"
        hard = cfg.teacher_selection == "hard"
        weights_used = list(self.controller.weights if active and hard else pending["weights_used"])
        last = pending["steps_done"] + 1 == pending["length"]
        qids = self.main_sampler.take(cfg.effective_batch_size)
        selected = (
            set(
                self.grounded_rng.choice(
                    len(qids), cfg.grounded_score_examples_per_step, replace=False
                ).tolist()
            )
            if active
            else set()
        )
        local = [(i, self.examples[q]) for i, q in enumerate(qids) if i % self.world == self.rank]
        rollouts = self.generate_many([ex for _, ex in local])
        pure = (
            [{n: torch.zeros_like(p) for n, p in self.params.items()} for _ in range(2)]
            if active and (last or hard)
            else None
        )
        cache = ReplayCache(cfg.replay_cache_dir) if active and last else None
        local_loss, stats, grounded = 0.0, [], []
        try:
            for (index, ex), tokens in zip(local, rollouts):
                full = self.score(ex, tokens, ex.reference_solution, "full_teacher")
                if active:
                    pool = self.pool(ex)
                    dropped, exposure = pool.sample(pending["alpha"], cfg.exposure_temperature, self.mask_rng)
                    reference = render_mask(ex, dropped, cfg.mask_teacher_include_final_answer)
                    mask = self.score(ex, tokens, reference, "mask_teacher")
                else:
                    pool, dropped, exposure, reference, mask = (
                        None,
                        frozenset(),
                        1.0,
                        ex.reference_solution,
                        full,
                    )
                alias = self.prompt(ex, reference) == self.prompt(ex, ex.reference_solution)
                gt_span, gt = {}, None
                if index in selected:
                    gt, gt_span = self.grounded_score(ex, reference)
                    grounded.append(gt)
                targets, divergence = [], 0.0
                with torch.no_grad():
                    for chunk in self.chunks(len(tokens)):
                        f, m = self.probabilities(full[chunk]), self.probabilities(mask[chunk])
                        if hard:
                            targets.append((f if weights_used[0] == 1 else m).cpu())
                        else:
                            targets.append((weights_used[0] * f + weights_used[1] * m).cpu())
                        if active:
                            logf = F.log_softmax(
                                full[chunk].to(self.device).float() / cfg.distillation_temperature, -1
                            )
                            logm = F.log_softmax(
                                mask[chunk].to(self.device).float() / cfg.distillation_temperature, -1
                            )
                            divergence += float((f * (logf - logm)).sum()) / len(tokens)
                    del f, m
                if cache is not None:
                    with self.timed("replay_cache_write"):
                        cache.add(ex, tokens, targets)
                if pure is not None:
                    value, gradients = self.expert_gradients(ex, tokens, full, mask, targets)
                    for acc, gs in zip(pure, gradients):
                        for name, g in gs.items():
                            acc[name].add_(g / cfg.effective_batch_size)
                    del gradients, gs, g
                else:
                    value = self.main_backward(ex, tokens, targets, len(local))
                local_loss += value / len(local)
                record = {
                    "step": self.step + 1,
                    "window_index": self.window_index,
                    "phase": pending["phase"],
                    "qid": ex.qid,
                    "alpha": pending["alpha"],
                    "policy_version": pending["policy_version"],
                    "weights_used": weights_used,
                    "teacher_selection": cfg.teacher_selection,
                    "answer_visibility": cfg.mask_teacher_include_final_answer,
                    "exposure": exposure,
                    "mask_token_percent": 100 * (1 - exposure),
                    "mask_step_percent": 100 * len(dropped) / len(ex.nodes),
                    "dropped": sorted(dropped),
                    "retained": [n for n in ex.node_ids if n not in dropped],
                    "candidate_count": sum(map(len, pool.groups.values())) if pool else 0,
                    "unique_levels": len(pool.groups) if pool else 0,
                    "pair_total": pool.pair_total if pool else 0,
                    "pair_used": pool.pair_used if pool else 0,
                    "branching": self.branching[ex.qid],
                    "alias_full": alias,
                    "all_reasoning_masked": len(dropped) == len(ex.nodes),
                    "full_context_tokens": len(self.prompt(ex, ex.reference_solution)),
                    "mask_context_tokens": len(self.prompt(ex, reference)),
                    "kl_full_mask": divergence,
                    "rollout_tokens": len(tokens),
                    "truncated": len(tokens) == cfg.main_max_new_tokens,
                    "grounded_selected": index in selected,
                    "grounded_log_probability": gt,
                    **gt_span,
                }
                self.samples.write(
                    {
                        **record,
                        "tokens": tokens,
                        "completion": self.tokenizer.decode(tokens),
                        "masked_reference": reference,
                    }
                )
                stats.append(record)
                del full, mask, targets
            gathered = self.gather({"loss": local_loss, "stats": stats, "grounded": grounded})
            all_stats = [r for rank in gathered for r in rank["stats"]]
            loss = float(np.mean([r["loss"] for r in gathered]))
            pending["sample_stats"].extend(all_stats)
            if active:
                pending["loss_observations"].append(loss)
                gt_all = [g for rank in gathered for g in rank["grounded"]]
                gt_value = (
                    float(np.mean(gt_all))
                    if len(gt_all) == cfg.grounded_score_examples_per_step
                    and all(g is not None and math.isfinite(g) for g in gt_all)
                    else None
                )
                pending["grounded_observations"].append(gt_value)
            duel = {"duel_valid": False, "duel_skip_reason": "full_phase"}
            valid_controls = sum(r["valid"] for r in pending["control"])
            if pure is not None:
                if self.world > 1:
                    for gs in pure:
                        for g in gs.values():
                            dist.all_reduce(g)
                if valid_controls < cfg.min_valid_controls:
                    duel["duel_skip_reason"] = "insufficient_valid_controls"
                elif all(r["alias_full"] for r in all_stats):
                    duel["duel_skip_reason"] = "all_duel_masks_alias_full"
                else:
                    with self.timed("virtual_duel"):
                        duel = virtual_duel(
                            self.model,
                            self.params,
                            self.optimizer,
                            *pure,
                            lambda: self.control_value(pending["control"])[0],
                            cfg.max_grad_norm,
                            cfg.numeric_epsilon,
                        )
                for name, parameter in self.params.items():
                    if hard:
                        parameter.grad = pure[0 if weights_used[0] == 1 else 1][name]
                    else:
                        parameter.grad = weights_used[0] * pure[0][name] + weights_used[1] * pure[1][name]
                del pure
            else:
                for p in self.params.values():
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    if self.world > 1:
                        dist.all_reduce(p.grad)
                        p.grad.div_(self.world)
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    list(self.params.values()), cfg.max_grad_norm if cfg.max_grad_norm else float("inf")
                )
            )
            if not math.isfinite(norm):
                raise RuntimeError("Nonfinite student gradient; no optimizer commit")
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            pending["steps_done"] += 1
            if cache is not None:
                pending["loss_observations"].append(cache.loss(self))
                self.count("replay_cache_bytes", cache.bytes)
            expert_feedback = None
            if hard:
                pending["step_weights"].append(weights_used)
            if active and hard:
                expert_feedback = self.controller.update_experts(duel) if self.rank == 0 else None
                expert_feedback, state = self.broadcast(
                    (expert_feedback, self.controller.state_dict()) if self.rank == 0 else None
                )
                if self.rank != 0:
                    self.controller.restore(state)
                pending["expert_feedback"].append({"step": self.step, **expert_feedback})
            feedback = None
            if active and last:
                reward = exposure_reward(
                    pending["loss_observations"],
                    pending["grounded_observations"],
                    cfg.gamma_discount,
                    cfg.lambda_gt,
                )
                if all(r["alias_full"] for r in pending["sample_stats"]):
                    reward.update(actor_valid=False, actor_skip_reason="all_window_masks_alias_full")
                if self.rank == 0:
                    if hard:
                        feedback = self.controller.finish(
                            pending,
                            reward,
                            expert_feedback,
                            update_ftrl=False,
                            actor_mask_weight=float(np.mean([w[1] for w in pending["step_weights"]])),
                        )
                    else:
                        feedback = self.controller.finish(pending, reward, duel)
                feedback, state = self.broadcast(
                    (feedback, self.controller.state_dict()) if self.rank == 0 else None
                )
                if self.rank != 0:
                    self.controller.restore(state)
            counters = self.gather(
                {"counts": self.counts, "timings": self.timings, "seconds": time.monotonic() - started}
            )
            counts = {
                k: sum(r["counts"].get(k, 0) for r in counters)
                for k in set().union(*(r["counts"] for r in counters))
            }
            for key, value in counts.items():
                pending["window_counts"][key] = pending["window_counts"].get(key, 0) + value
            for key in set().union(*(r["timings"] for r in counters)):
                pending["window_timings"][key] = pending["window_timings"].get(key, 0.0) + max(
                    r["timings"].get(key, 0) for r in counters
                )
            seconds = max(r["seconds"] for r in counters)
            pending["wall_seconds"] += seconds
            result = {
                "step": self.step,
                "phase": pending["phase"],
                "window_index": self.window_index,
                "loss": loss,
                "grad_norm": norm,
                "learning_rate": lr,
                "alpha": pending["alpha"],
                "weights_used": weights_used,
                "latent_weights": weights_used if hard and active else list(pending["latent_weights_used"]),
                "teacher_selection": cfg.teacher_selection,
                "teacher_used": "FULL"
                if weights_used == [1, 0]
                else "MASK"
                if weights_used == [0, 1]
                else "mixture",
                "mask_token_percent": float(np.mean([r["mask_token_percent"] for r in all_stats])),
                "mask_step_percent": float(np.mean([r["mask_step_percent"] for r in all_stats])),
                "kl_full_mask": float(np.mean([r["kl_full_mask"] for r in all_stats])),
                "teacher_alias_rate": float(np.mean([r["alias_full"] for r in all_stats])),
                "grounded_score": pending["grounded_observations"][-1] if active else None,
                "valid_control_count": valid_controls,
                "generated_tokens": sum(r["rollout_tokens"] for r in all_stats),
                "truncation_rate": float(np.mean([r["truncated"] for r in all_stats])),
                "effective_batch_size": cfg.effective_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "generation_backend": cfg.generation_backend,
                "counts": counts,
                "wall_seconds": seconds,
            }
            if expert_feedback is not None:
                result["expert_feedback"] = expert_feedback
            if last:
                if feedback is None:
                    feedback = {
                        "actor_valid": False,
                        "duel_valid": False,
                        "actor_skip_reason": "full_phase",
                        "duel_skip_reason": "full_phase",
                        "weights_next": list(self.controller.weights),
                        "controller_updated": False,
                        "ftrl_updated": False,
                    }
                result["window_feedback"] = feedback
                window = {
                    **feedback,
                    "phase": pending["phase"],
                    "window_index": self.window_index,
                    "end_step": self.step,
                    "length": pending["length"],
                    "state_used": pending["state"],
                    "alpha": pending["alpha"],
                    "weights_used": pending["weights_used"],
                    "latent_weights_used": pending["latent_weights_used"],
                    "beta_a": pending["beta_a"],
                    "beta_b": pending["beta_b"],
                    "beta_mean": pending["beta_a"] / (pending["beta_a"] + pending["beta_b"]),
                    "beta_concentration": pending["beta_a"] + pending["beta_b"],
                    "valid_control_count": valid_controls,
                    "control_exclusions": [r["reason"] for r in pending["control"] if not r["valid"]],
                    "answer_visibility": cfg.mask_teacher_include_final_answer,
                    "exposure_mean": float(np.mean([r["exposure"] for r in pending["sample_stats"]])),
                    "exposure_min": min(r["exposure"] for r in pending["sample_stats"]),
                    "exposure_max": max(r["exposure"] for r in pending["sample_stats"]),
                    "teacher_alias_rate": float(np.mean([r["alias_full"] for r in pending["sample_stats"]])),
                    "kl_full_mask": float(np.mean([r["kl_full_mask"] for r in pending["sample_stats"]])),
                    "full_context_tokens_mean": float(
                        np.mean([r["full_context_tokens"] for r in pending["sample_stats"]])
                    ),
                    "mask_context_tokens_mean": float(
                        np.mean([r["mask_context_tokens"] for r in pending["sample_stats"]])
                    ),
                    "branching_rate": float(np.mean([r["branching"] for r in pending["sample_stats"]])),
                    "mask_diversity": len({(r["qid"], tuple(r["dropped"])) for r in pending["sample_stats"]}),
                    "grounded_reference_fraction_mean": float(
                        np.mean(
                            [
                                r.get("reference_fraction", 0.0)
                                for r in pending["sample_stats"]
                                if r["grounded_selected"]
                            ]
                        )
                    )
                    if active
                    else None,
                    "grounded_reference_truncation_rate": float(
                        np.mean(
                            [
                                r.get("reference_truncated", False)
                                for r in pending["sample_stats"]
                                if r["grounded_selected"]
                            ]
                        )
                    )
                    if active
                    else None,
                    "candidate_count_mean": float(
                        np.mean([r["candidate_count"] for r in pending["sample_stats"]])
                    ),
                    "unique_levels_mean": float(
                        np.mean([r["unique_levels"] for r in pending["sample_stats"]])
                    ),
                    "answer_only_mask_rate": float(
                        np.mean([r["all_reasoning_masked"] for r in pending["sample_stats"]])
                    )
                    if cfg.mask_teacher_include_final_answer
                    else 0.0,
                    "question_only_mask_rate": float(
                        np.mean([r["all_reasoning_masked"] for r in pending["sample_stats"]])
                    )
                    if not cfg.mask_teacher_include_final_answer
                    else 0.0,
                    "dataset_eligibility": {k: v for k, v in self.metadata.items() if k != "invalid_rows"},
                    "counts": pending["window_counts"],
                    "timing_seconds_rank_max": pending["window_timings"],
                    "wall_seconds": pending["wall_seconds"],
                }
                if hard:
                    window.update(
                        teacher_selection="hard",
                        expert_refresh_steps=1,
                        step_weights=pending["step_weights"],
                        expert_feedback=pending["expert_feedback"],
                        expert_updates_in_window=sum(f["ftrl_updated"] for f in pending["expert_feedback"]),
                    )
                if self.rank == 0:
                    self.windows.write(window)
                self.window_index += 1
                self.pending = None
            if self.rank == 0:
                self.metrics.write(result)
            return result
        finally:
            if cache is not None:
                cache.close()

    def train(self, resume=None):
        from .checkpoint import restore_checkpoint, save_checkpoint

        if resume:
            restore_checkpoint(self, resume)
        with tqdm(
            total=self.config.total_updates,
            initial=self.step,
            dynamic_ncols=True,
            unit="step",
            disable=self.rank != 0,
        ) as progress:
            self.progress = progress
            try:
                while self.step < self.config.total_updates:
                    result = self.run_update()
                    if self.rank == 0:
                        progress.update()
                        if self.config.verbose_runtime:
                            progress.write(str(result))
                        keys = (
                            "step",
                            "phase",
                            "loss",
                            "grad_norm",
                            "learning_rate",
                            "alpha",
                            "mask_token_percent",
                            "mask_step_percent",
                            "grounded_score",
                        )
                        log = {
                            k: round(result[k], 6) if isinstance(result[k], float) else result[k]
                            for k in keys
                        }
                        log["w_F"], log["w_M"] = [round(x, 4) for x in result["weights_used"]]
                        log["learning_rate"] = result["learning_rate"]
                        if self.config.teacher_selection == "hard":
                            log["teacher"] = result["teacher_used"]
                            if "expert_feedback" in result:
                                expert = result["expert_feedback"]
                                log["teacher_next"] = "MASK" if expert["weights_next"][1] == 1 else "FULL"
                                log["delta"] = expert.get("delta")
                                log["D"] = round(expert["D"], 6)
                                log["ftl_updated"] = expert["ftrl_updated"]
                                log["ftl_skip"] = expert["duel_skip_reason"]
                        log["controls"] = (
                            f"{result['valid_control_count']}/{self.config.control_batch_size}"
                            if result["phase"] == "adaptive"
                            else "-"
                        )
                        progress.write(str(log))
                        if "window_feedback" in result and result["phase"] == "adaptive":
                            f = result["window_feedback"]
                            progress.write(
                                str(
                                    {
                                        "window": result["window_index"],
                                        "G_lp": f.get("G_lp"),
                                        "G_gt": f.get("G_gt"),
                                        "R": f.get("R"),
                                        "delta": f.get("delta"),
                                        "delta_scaled": f.get("delta_scaled"),
                                        "w_M_next": round(f["weights_next"][1], 4),
                                        "actor_updated": f["controller_updated"],
                                        "actor_skip": f["actor_skip_reason"],
                                        "duel_skip": f["duel_skip_reason"],
                                    }
                                )
                            )
                    if self.step % self.config.save_every == 0 or self.step == self.config.total_updates:
                        save_checkpoint(self)
            finally:
                self.progress = None
