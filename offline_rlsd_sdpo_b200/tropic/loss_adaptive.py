"""TROPIC-ADAPTIVE: Eq. 18 (with Eq. 18's mu-hinge, as in loss_k.py) plus
Eq. 20's adaptive trust-region radius. epsilon is no longer a fixed scalar:
it is recomputed from the privileged gap G_k (Eq. 19) via the clipped
linear schedule

    epsilon_k = eps_min + (eps_max - eps_min) * clip((G_k - G_min)/(G_max - G_min), 0, 1)

and that SAME epsilon_k is then used both for the projection (Eq. 13-16,
where the fixed epsilon used to go) and for the hinge threshold in Eq. 18
(same substitution, since the paper never introduces a separate symbol for
the two - they are the same epsilon, now iteration-dependent).

DELIBERATE SIMPLIFICATION vs Algorithm 1's pseudocode: the paper computes
G_k/epsilon_k ONCE per outer iteration, aggregated over the whole batch of
that iteration's roll-outs, BEFORE looping over tokens (lines 6-7 precede
the per-token loop at lines 8-15). This codebase's train_run processes one
micro-example at a time (sequential forward/backward per example inside
the effective_batch loop, no separate batched-aggregation pass), so this
file computes G_k/epsilon_k PER MICRO-EXAMPLE instead of once per outer
iteration - each example's own privileged gap sets its own radius, rather
than one shared radius for the whole batch. This preserves the mechanism's
intent (throttle the radius when the gap is small, widen it when large) at
finer granularity than the pseudocode describes. Flagging this explicitly
rather than silently approximating it.

Kept in a separate module - loss.py and loss_k.py are both untouched.
"""
from __future__ import annotations

import torch
from torch import Tensor

from tropic.primitives import (
    apply_sparse_mask,
    build_shared_topk_mask,
    debias_logits,
    kl_categorical,
    normalize_log,
    project_to_kl_ball,
    skewed_reverse_kl,
    total_variation,
)


def adaptive_epsilon(G_k: float, epsilon_min: float, epsilon_max: float, G_min: float, G_max: float) -> float:
    """Eq. 20's clipped linear schedule, as a plain float (G_k has already
    been detached/`.item()`-ed by the caller - epsilon_k must never carry
    gradient, same contract as epsilon itself in loss.py/loss_k.py)."""
    if G_max <= G_min:
        raise ValueError(f"G_max must be > G_min, got G_min={G_min}, G_max={G_max}")
    frac = (G_k - G_min) / (G_max - G_min)
    frac = min(max(frac, 0.0), 1.0)
    return epsilon_min + (epsilon_max - epsilon_min) * frac


def tropic_adaptive_loss(
    log_student: Tensor,
    log_teacher: Tensor,
    log_old: Tensor,
    generated_ids: Tensor,
    epsilon_min: float,
    epsilon_max: float,
    G_min: float,
    G_max: float,
    beta: float,
    top_k: int,
    default_mass: float,
    mu: float,
    alpha: float = 0.0,
    teacher_context_only_logits: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """Same contract as tropic_k_loss, except `epsilon` is replaced by
    `epsilon_min`/`epsilon_max`/`G_min`/`G_max` and recomputed per call from
    this example's own G_k (Eq. 19-20) instead of being a fixed argument.

    Returns (loss, s, hinge_mean, G_k, epsilon_k):
      - hinge_mean: E_t[[KL(student||old) - epsilon_k]_+], pre-mu, like
        tropic_k_loss's third return value.
      - G_k: this example's privileged gap (Eq. 19), detached scalar tensor
        - log this to see the raw signal driving epsilon_k.
      - epsilon_k: the actual radius used this call (plain float) - log
        this to see the schedule's trajectory over training.
    """
    if alpha > 0.0:
        assert teacher_context_only_logits is not None, (
            "alpha>0 (TROPIC-G debiasing) requires teacher_context_only_logits"
        )
        assert teacher_context_only_logits.requires_grad is False, (
            "teacher_context_only_logits must be cached/detached (Section 4.5), same contract as log_teacher"
        )
        teacher_context_only_logits = teacher_context_only_logits.detach()
        log_teacher = normalize_log(debias_logits(log_teacher, teacher_context_only_logits, alpha))

    # Eq. 19: G_k on the FULL (unsparsified) checkpoint/debiased-teacher
    # pair - same convention already used elsewhere in this codebase for
    # the (previously diagnostic-only) privileged_gap() call, now actually
    # consumed to drive epsilon_k instead of only being logged.
    G_k_tensor = total_variation(log_old, log_teacher).mean().detach()
    epsilon_k = adaptive_epsilon(G_k_tensor.item(), epsilon_min, epsilon_max, G_min, G_max)

    mask = build_shared_topk_mask(log_old, generated_ids, top_k)
    log_student_sp = apply_sparse_mask(log_student, mask, default_mass)
    log_teacher_sp = apply_sparse_mask(log_teacher, mask, default_mass)
    log_old_sp = apply_sparse_mask(log_old, mask, default_mass)

    log_pi_eps, s = project_to_kl_ball(log_teacher_sp, log_old_sp, epsilon_k)
    log_pi_eps = log_pi_eps.detach()
    assert log_pi_eps.requires_grad is False

    per_token = skewed_reverse_kl(log_student_sp, log_pi_eps, beta)
    distill_loss = per_token.mean()

    kl_student_old = kl_categorical(log_student_sp, log_old_sp.detach())
    hinge = torch.clamp(kl_student_old - epsilon_k, min=0.0)
    loss = distill_loss + mu * hinge.mean()

    return loss, s, hinge.mean().detach(), G_k_tensor, epsilon_k
