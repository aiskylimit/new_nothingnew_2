"""Exact isolated AdamW branches on LoRA weights; no backbone copies."""

import copy
import math

import numpy as np
import torch

from .rng import rng_state, restore_rng


def virtual_duel(model, params, optimizer, full_grads, mask_grads, evaluate, max_grad_norm, epsilon):
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("Virtual duel requires the actual AdamW optimizer recipe")
    result = {"duel_valid": False, "duel_skip_reason": "nonfinite_virtual_gradient_or_state"}
    if any(not torch.isfinite(g).all() for gs in (full_grads, mask_grads) for g in gs.values()):
        return result
    saved_weights = {n: p.detach().clone() for n, p in params.items()}
    saved_buffers = [
        (
            m,
            dict(m._buffers),
            {n: None if b is None else b.detach().clone() for n, b in m._buffers.items()},
            set(m._non_persistent_buffers_set),
        )
        for m in model.modules()
    ]
    saved_modes = [(m, m.training) for m in model.modules()]
    saved_rng = rng_state()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    names = {id(p): n for n, p in params.items()}
    norms, losses = [], []

    def restore_buffers():
        for module, bindings, data, nonpersistent in saved_buffers:
            for name, buffer in bindings.items():
                if buffer is not None:
                    buffer.copy_(data[name])
            module._buffers.clear()
            module._buffers.update(bindings)
            module._non_persistent_buffers_set = set(nonpersistent)

    try:
        model.eval()
        with torch.no_grad():
            losses.append(evaluate())
        for gradients in (full_grads, mask_grads):
            clones = {n: torch.nn.Parameter(w.clone()) for n, w in saved_weights.items()}
            groups = [
                {
                    **{k: v for k, v in group.items() if k != "params"},
                    "params": [clones[names[id(p)]] for p in group["params"]],
                }
                for group in optimizer.param_groups
            ]
            branch = torch.optim.AdamW(groups)
            branch.load_state_dict(copy.deepcopy(saved_optimizer))
            for name, parameter in clones.items():
                parameter.grad = gradients[name].detach().clone()
            norm = torch.nn.utils.clip_grad_norm_(
                list(clones.values()), max_grad_norm if max_grad_norm else float("inf")
            )
            norms.append(float(norm))
            branch.step()
            tensors = [p.detach() for p in clones.values()]
            tensors += [v for state in branch.state.values() for v in state.values() if torch.is_tensor(v)]
            if any(not torch.isfinite(t).all() for t in tensors):
                return result
            with torch.no_grad():
                for name, parameter in params.items():
                    parameter.copy_(clones[name])
                # Every branch sees the original buffers/modes/RNG, not the previous branch.
                restore_buffers()
                restore_rng(saved_rng)
                model.eval()
                losses.append(evaluate())
            del branch, clones, groups, tensors
        if not all(x is not None and math.isfinite(x) for x in (*losses, *norms)) or losses[0] < 0:
            return {**result, "L_0": losses[0], "L_F": losses[1], "L_M": losses[2]}
        base, full, mask = losses
        resolution = float(abs(np.spacing(np.float32(max(losses)))))
        return {
            "duel_valid": True,
            "duel_skip_reason": "",
            "L_0": base,
            "L_F": full,
            "L_M": mask,
            "delta": (full - mask) / (base + epsilon),
            "gain_F": (base - full) / (base + epsilon),
            "gain_M": (base - mask) / (base + epsilon),
            "g_F_norm": norms[0],
            "g_M_norm": norms[1],
            "loss_contrast": full - mask,
            "loss_resolution": resolution,
            "tiny_or_zero_contrast": abs(full - mask) <= resolution,
            "virtual_optimizer": "exact_cloned_adamw",
            "virtual_learning_rates": [g["lr"] for g in optimizer.param_groups],
            "virtual_max_grad_norm": max_grad_norm,
        }
    finally:
        with torch.no_grad():
            for name, parameter in params.items():
                parameter.copy_(saved_weights[name])
            restore_buffers()
        for module, training in saved_modes:
            module.training = training
        restore_rng(saved_rng)
        # The live optimizer and .grad were never used by either virtual branch.
