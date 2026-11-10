"""TROPIC-K: Eq. 18 in FULL, with the mu-weighted hinge certificate
(Corollary 1) activated on top of TROPIC-G's debiased/projected distillation
term. loss.py hardcodes mu=0 by design (see its own docstring and
Proposition 5's justification for why mu=0 is already sound on average); this
module is kept fully separate - it does not import from loss.py and loss.py
is never touched - so existing TROPIC-P/TROPIC-G/OPSD/RLSD/SDPO runs and
scripts are unaffected.

Eq. 18:
    J = E_{x,o}[ (1/|o|) sum_t ( D^(beta)(pi^S_t || floor(pi^eps_t))
                                  + mu * [KL(pi^S_t || pi^old_t) - epsilon]_+ ) ]

The first term is identical to tropic_p_loss/tropic_g (alpha=0 or alpha>0).
The second term is new: a per-token hinge on the STUDENT's own realized KL
drift from the checkpoint, active only on positions where the student has
actually left the epsilon-ball (Corollary 1: "the hinge upgrades the bound
to a certificate"). It carries gradient through log_student - unlike
kl_student_old_drift logged elsewhere in this codebase (train.py,
run_full_experiment*.py), which is computed under torch.no_grad() purely as
a diagnostic and never enters any loss.
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
)


def tropic_k_loss(
    log_student: Tensor,
    log_teacher: Tensor,
    log_old: Tensor,
    generated_ids: Tensor,
    epsilon: float,
    beta: float,
    top_k: int,
    default_mass: float,
    mu: float,
    alpha: float = 0.0,
    teacher_context_only_logits: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Drop-in analogue of tropic_p_loss with Eq. 18's hinge term added.

    Same contract as tropic_p_loss for every shared argument (alpha=0
    recovers TROPIC-P's distillation term, alpha>0 recovers TROPIC-G's - the
    debiasing/projection logic below is duplicated from loss.py rather than
    imported so this file has zero coupling to it). `mu` is new: the Eq. 18
    hinge weight. mu=0.0 makes the returned `loss` numerically identical to
    tropic_p_loss's (up to the extra hinge_mean return value).

    Returns (loss, s, hinge_mean): hinge_mean is detached, logging-only - it
    is E_t[[KL(pi^S_t || pi^old_t) - epsilon]_+] BEFORE multiplying by mu, so
    it reads directly as "how much the certificate is firing", independent of
    whatever mu is currently set to.
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

    # Remark 4: one shared top-K mask for every distribution in Eq. 18,
    # including the hinge term below - this is what makes the
    # sparsification error bound apply to the objective as a whole.
    mask = build_shared_topk_mask(log_old, generated_ids, top_k)

    log_student_sp = apply_sparse_mask(log_student, mask, default_mass)
    log_teacher_sp = apply_sparse_mask(log_teacher, mask, default_mass)
    log_old_sp = apply_sparse_mask(log_old, mask, default_mass)

    log_pi_eps, s = project_to_kl_ball(log_teacher_sp, log_old_sp, epsilon)
    log_pi_eps = log_pi_eps.detach()
    assert log_pi_eps.requires_grad is False

    per_token = skewed_reverse_kl(log_student_sp, log_pi_eps, beta)
    distill_loss = per_token.mean()

    # Eq. 18's second term: mu * [KL(pi^S_t || pi^old_t) - epsilon]_+. Plain
    # (unskewed) KL, student first, on the same sparsified support as above.
    # log_old_sp already carries no grad (checkpoint); log_student_sp
    # carries grad through log_student, so this term actively pulls the
    # student back once it has left the ball, rather than merely reporting
    # the drift the way kl_student_old_drift does.
    kl_student_old = kl_categorical(log_student_sp, log_old_sp.detach())
    hinge = torch.clamp(kl_student_old - epsilon, min=0.0)
    loss = distill_loss + mu * hinge.mean()

    return loss, s, hinge.mean().detach()
