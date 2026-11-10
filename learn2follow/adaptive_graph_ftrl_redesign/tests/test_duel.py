import copy

import pytest
import torch

from adaptive_graph_ftrl_redesign.duel import virtual_duel


def equal_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("primed", [False, True])
def test_exact_virtual_adamw_and_full_restore(primed):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    model.register_buffer("counter", torch.tensor(2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.013, betas=(0.8, 0.95), weight_decay=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 1 / (n + 1))
    if primed:
        model(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    params = dict(model.named_parameters())
    full = {n: torch.randn_like(p) for n, p in params.items()}
    mask = {n: torch.randn_like(p) for n, p in params.items()}
    old_weights = copy.deepcopy(model.state_dict())
    old_opt = copy.deepcopy(optimizer.state_dict())
    old_sched = copy.deepcopy(scheduler.state_dict())
    old_grads = {n: None if p.grad is None else p.grad.clone() for n, p in params.items()}
    x = torch.tensor([[1.0, -2.0, 0.5]])
    expected = []
    for grads in (full, mask):
        real = copy.deepcopy(model)
        opt = torch.optim.AdamW(real.parameters(), lr=0.013)
        opt.load_state_dict(copy.deepcopy(old_opt))
        for n, p in real.named_parameters():
            p.grad = grads[n].clone()
        torch.nn.utils.clip_grad_norm_(real.parameters(), 0.1)
        opt.step()
        expected.append(float(real(x).square().mean().detach()))
    rng = torch.get_rng_state().clone()

    def evaluate():
        assert not model.training
        assert model.counter.item() == 2
        model.counter.add_(1)
        model.register_buffer("counter", model.counter + 1, persistent=False)
        torch.rand(3)
        return float(model(x).square().mean())

    info = virtual_duel(model, params, optimizer, full, mask, evaluate, 0.1, 1e-12)
    assert info["duel_valid"]
    assert [info["L_F"], info["L_M"]] == expected
    equal_tree(old_weights, model.state_dict())
    equal_tree(old_opt, optimizer.state_dict())
    equal_tree(old_sched, scheduler.state_dict())
    for n, p in params.items():
        equal_tree(old_grads[n], p.grad)
    assert model.training and torch.equal(rng, torch.get_rng_state())


def test_invalid_gradient_no_mutation():
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.AdamW(model.parameters())
    params = dict(model.named_parameters())
    gs = {n: torch.full_like(p, float("nan")) for n, p in params.items()}
    before = copy.deepcopy(model.state_dict())
    assert not virtual_duel(model, params, opt, gs, gs, lambda: 1, 0.1, 1e-12)["duel_valid"]
    equal_tree(before, model.state_dict())


def test_exception_during_virtual_evaluation_restores_weights_and_buffers():
    model = torch.nn.Linear(1, 1)
    model.register_buffer("counter", torch.tensor(2))
    opt = torch.optim.AdamW(model.parameters())
    params = dict(model.named_parameters())
    gs = {n: torch.ones_like(p) for n, p in params.items()}
    before = copy.deepcopy(model.state_dict())
    calls = 0

    def evaluate():
        nonlocal calls
        calls += 1
        if calls == 2:
            model.register_buffer("counter", torch.tensor(10))
            raise RuntimeError("evaluation failed")
        return 1.0

    with pytest.raises(RuntimeError, match="evaluation failed"):
        virtual_duel(model, params, opt, gs, gs, evaluate, 0.1, 1e-12)
    equal_tree(before, model.state_dict())
    assert model.training and not opt.state
