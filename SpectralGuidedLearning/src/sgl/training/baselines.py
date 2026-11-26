"""ASFT and VCORE, two token-reweighting baselines, on top of the masked SFT objective.

Both keep the data, mask and normalizer Z (supervised tokens of the optimizer step) of the other
SFT arms; only the per-token coefficient changes.

ASFT (arXiv 2509.23753, github.com/zhuchichi56/ASFT, `EnhancedTrainer` mode "asft"):
    l_t = sg(p_t) * (-log p_t) + lambda * KL(pi_base(.|t) || pi_theta(.|t))
with pi_base the frozen pretrained student.

VCORE (arXiv 2510.27462, github.com/coder-gx/VCORE, `vcore_loss` + `_run_anchor_window`):
    1. g = clipped gradient of uniform CE on a random batch B' (as many samples as one update)
    2. gain_t = ce_t(theta) - ce_t(theta - epsilon * g)
    3. w = softmax(gain / (epsilon * temperature)) over a sample's supervised tokens, times their count
    4. L = min(1, L_uniform / L_weighted) * L_weighted
The formulas follow the released code (the paper writes exp(tau * s); the code divides by tau * epsilon).

Where the port differs from the VCORE release, both forced by full fine-tuning (upstream trains LoRA
adapters only):
  - the probe step moves every trainable parameter;
  - ce_t(theta) and ce_t(theta - epsilon * g) come from two fp32 forwards. A step of norm epsilon spread
    over 1.8e9 weights is far below bf16 resolution, so bf16 forwards would give gain = 0 everywhere.
    The weights stay fp32 (no DeepSpeed); the training forward itself runs under bf16 autocast.
"""

import contextlib
import random

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback

from sgl.training.losses import IGNORE_INDEX
from sgl.training.utils import compensate_global_token_mean

# positions per chunk: a 32k-token trace times a 152k vocabulary does not fit several float32 copies
CHUNK = 2048


def _anchored_chunk(logits, ref_logits, labels, kl_lambda):
    valid = labels != IGNORE_INDEX
    log_q = F.log_softmax(logits[valid].float(), dim=-1)
    nll = -log_q.gather(1, labels[valid].unsqueeze(1)).squeeze(1)
    log_p_ref = F.log_softmax(ref_logits[valid].float(), dim=-1)
    kl = (log_p_ref.exp() * (log_p_ref - log_q)).sum(dim=-1)
    return (torch.exp(-nll.detach()) * nll + kl_lambda * kl).sum()


def anchored_loss_sum(logits, ref_logits, labels, kl_lambda: float, chunk: int = CHUNK) -> torch.Tensor:
    """Sum of the ASFT per-token loss over supervised positions. All inputs unshifted (B, T[, V])."""
    shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    shift_ref = ref_logits[:, :-1].reshape(-1, ref_logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    total = shift_logits.new_zeros((), dtype=torch.float32)
    for part, ref, target in zip(shift_logits.split(chunk), shift_ref.split(chunk), shift_labels.split(chunk)):
        if (target == IGNORE_INDEX).all():
            continue
        # recompute in backward: only one chunk's log-softmax is alive at a time
        if part.requires_grad:
            total = total + checkpoint(_anchored_chunk, part, ref, target, kl_lambda, use_reentrant=False)
        else:
            total = total + _anchored_chunk(part, ref, target, kl_lambda)
    # a batch without supervised tokens still has to produce a differentiable zero
    return total + shift_logits[:1].sum() * 0.0


def _ce_chunk(logits, labels):
    return F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="none")


def per_token_ce(logits, labels, chunk: int = CHUNK) -> torch.Tensor:
    """(B, T-1) cross-entropy of every supervised position, 0 elsewhere, from unshifted (B, T[, V]) inputs."""
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    parts = shift_logits.reshape(-1, shift_logits.size(-1)).split(chunk)
    targets = shift_labels.reshape(-1).split(chunk)
    losses = [
        checkpoint(_ce_chunk, part, target, use_reentrant=False)
        if part.requires_grad else _ce_chunk(part, target)
        for part, target in zip(parts, targets)
    ]
    return torch.cat(losses).view_as(shift_labels)


def vcore_weights(gain: torch.Tensor, valid: torch.Tensor, probe_lr: float, temperature: float) -> torch.Tensor:
    """Per-sample softmax of gain / (probe_lr * temperature), rescaled to mean 1 over supervised tokens."""
    scores = (gain / (probe_lr * temperature)).masked_fill(~valid, float("-inf"))
    peak = scores.max(dim=-1, keepdim=True).values
    exp = torch.where(valid, (scores - peak).exp(), torch.zeros_like(scores))
    weights = exp / exp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights * valid.sum(dim=-1, keepdim=True).clamp_min(1)


def vcore_loss(logits, labels, weights, denominator) -> tuple[torch.Tensor, float]:
    """Weighted CE and the variance-control scale min(1, uniform / weighted) applied to it."""
    ce = per_token_ce(logits, labels)
    valid = (labels[:, 1:] != IGNORE_INDEX).float()
    denominator = torch.as_tensor(denominator, device=ce.device).float().clamp_min(1.0)
    weighted = (weights * ce * valid).sum() / denominator
    uniform = (ce.detach() * valid).sum() / denominator
    scale = (uniform / weighted.detach().clamp_min(1e-12)).clamp(max=1.0)
    return scale * weighted, float(scale)


class AnchoredMixin:
    """ASFT objective for MaskedSFTTrainer."""

    def __init__(self, *args, reference_model, kl_lambda: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model.eval().requires_grad_(False)
        self.kl_lambda = kl_lambda
        self._reference_on_device = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        if "loss_weights" in inputs:
            raise ValueError("ASFT weights every supervised token itself; use unweighted (vanilla) data")
        if not self._reference_on_device:
            self.reference_model.to(self.args.device)
            self._reference_on_device = True
        outputs = model(**inputs)
        with torch.no_grad():
            ref_logits = self.reference_model(**inputs).logits
        if num_items_in_batch is None:
            denominator = (labels[:, 1:] != IGNORE_INDEX).sum()
        else:
            denominator = num_items_in_batch
        total = anchored_loss_sum(outputs.logits, ref_logits, labels, self.kl_lambda)
        loss = total / torch.as_tensor(denominator, device=total.device).float().clamp_min(1.0)
        loss = compensate_global_token_mean(
            loss,
            average_tokens_across_devices=self.args.average_tokens_across_devices,
            num_items_in_batch=num_items_in_batch,
            num_processes=self.accelerator.num_processes,
        )
        return (loss, outputs) if return_outputs else loss

    def method_summary(self) -> dict:
        return {"baseline": "asft", "kl_lambda": self.kl_lambda}


class _ProbeCallback(TrainerCallback):
    """Computes the probe direction once per optimizer step, while no gradient is accumulated yet."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_begin(self, args, state, control, **kwargs):
        self.trainer.compute_probe_direction()


class VCoreMixin:
    """VCORE objective for MaskedSFTTrainer (single process, fp32 weights, no DeepSpeed)."""

    def __init__(
        self, *args, probe_epsilon: float, temperature: float, probe_batches: int | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.args.world_size > 1 or self.args.n_gpu > 1 or self.args.deepspeed:
            raise ValueError("VCORE runs on one GPU without DeepSpeed: the probe step edits the weights")
        if any(p.dtype != torch.float32 for p in self.model.parameters() if p.requires_grad):
            raise ValueError("VCORE needs fp32 weights (--model-dtype float32)")
        self.probe_epsilon = probe_epsilon
        self.temperature = temperature
        # upstream draws as many samples for B' as one optimizer step holds
        self.probe_batches = probe_batches or self.args.gradient_accumulation_steps
        self._params = None
        self._direction = None
        self._backup = None
        self._stats = {"microbatches": 0, "abs_gain": 0.0, "nonzero_gain": 0.0,
                       "effective_token_fraction": 0.0, "max_weight": 0.0, "scale": 0.0}
        self.add_callback(_ProbeCallback(self))

    def _autocast(self):
        """bf16 for the training forward and the probe gradient; scoring forwards stay outside it."""
        if self.args.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _probe_indices(self) -> list[int]:
        rng = random.Random()
        indices = []
        for batch in range(self.probe_batches):
            rng.seed((self.state.epoch or 0) * 10_000_003 + batch * 97)  # upstream's seeding
            indices.append(rng.randrange(len(self.train_dataset)))
        return indices

    def compute_probe_direction(self) -> None:
        model = self.model
        model.train()
        if self._params is None:
            self._params = [p for p in model.parameters() if p.requires_grad]
        batches = [
            self._prepare_inputs(self.data_collator([self.train_dataset[i]])) for i in self._probe_indices()
        ]
        total = sum(int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in batches)
        for batch in batches:
            labels = batch.pop("labels")
            batch.pop("loss_weights", None)
            with self._autocast():
                logits = model(**batch).logits
            (per_token_ce(logits, labels).sum() / max(total, 1)).backward()
        torch.nn.utils.clip_grad_norm_(self._params, self.args.max_grad_norm)
        self._direction = [None if p.grad is None else p.grad.detach().clone() for p in self._params]
        for param in self._params:
            param.grad = None

    @contextlib.contextmanager
    def _probed_weights(self):
        """theta - epsilon * g inside the block, theta restored bit for bit afterwards."""
        with torch.no_grad():
            if self._backup is None:
                self._backup = [torch.empty_like(p) for p in self._params]
            for param, backup, grad in zip(self._params, self._backup, self._direction):
                backup.copy_(param)
                if grad is not None:
                    param.sub_(grad, alpha=self.probe_epsilon)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, backup in zip(self._params, self._backup):
                    param.copy_(backup)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        if "loss_weights" in inputs:
            raise ValueError("VCORE weights every supervised token itself; use unweighted (vanilla) data")
        if self._direction is None:
            raise RuntimeError("no probe direction: the on_step_begin callback did not run")
        with torch.no_grad():
            current = per_token_ce(model(**inputs).logits, labels)
            with self._probed_weights():
                probed = per_token_ce(model(**inputs).logits, labels)
        valid = labels[:, 1:] != IGNORE_INDEX
        gain = (current - probed) * valid
        weights = vcore_weights(gain, valid, self.probe_epsilon, self.temperature)
        with self._autocast():
            outputs = model(**inputs)
        denominator = num_items_in_batch if num_items_in_batch is not None else valid.sum()
        loss, scale = vcore_loss(outputs.logits, labels, weights, denominator)
        self._record(gain[valid], weights[valid], scale)
        return (loss, outputs) if return_outputs else loss

    def _record(self, gain, weights, scale) -> None:
        if not gain.numel():
            return
        stats = self._stats
        stats["microbatches"] += 1
        stats["abs_gain"] += float(gain.abs().mean())
        stats["nonzero_gain"] += float((gain != 0).float().mean())
        # 1.0 = uniform weighting, near 0 = a few tokens carry the sample
        stats["effective_token_fraction"] += float(weights.sum() ** 2 / (weights.numel() * (weights**2).sum()))
        stats["max_weight"] += float(weights.max())
        stats["scale"] += scale
        if stats["microbatches"] % 50 == 1:
            print(f"vcore weights, mean over {stats['microbatches']} samples: {self._mean_stats()}", flush=True)

    def _mean_stats(self) -> dict:
        count = max(self._stats["microbatches"], 1)
        return {k: round(v / count, 6) for k, v in self._stats.items() if k != "microbatches"}

    def method_summary(self) -> dict:
        return {
            "baseline": "vcore", "epsilon": self.probe_epsilon, "temperature": self.temperature,
            "probe_batches": self.probe_batches, "weight_stats": self._mean_stats(),
        }
