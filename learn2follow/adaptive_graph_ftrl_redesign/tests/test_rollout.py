"""Check snapshot export, sleep and RNG preservation without CUDA."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from adaptive_graph_ftrl_redesign.rollout import VLLMRollouts
from .fixtures import tiny_trainer
from .rollout_fixture import FakeEngine, request


def backend(trainer):
    return VLLMRollouts(trainer, (FakeEngine, lambda **kw: SimpleNamespace(**kw), request))

def test_current_snapshot_batch_and_sleep(tmp_path):
    trainer = tiny_trainer(tmp_path)
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    rng = torch.get_rng_state().clone()
    rollout = backend(trainer)
    assert torch.equal(rng, torch.get_rng_state())
    first = rollout.generate([[3, 4], [3, 5]], 5)
    assert first == [[14, 2], [14, 2]]
    assert rollout.engine.asleep
    assert trainer.counts["rollout_adapter_sync_calls"] == 1
    weights = load_file(str(rollout.path / "adapter_model.safetensors"))
    rollout.generate([[3, 4]], 4)
    assert trainer.counts["rollout_adapter_sync_calls"] == 1
    for name, p in trainer.model.named_parameters():
        assert torch.equal(before[name], p)
    with torch.no_grad():
        next(iter(trainer.params.values())).add_(0.1)
    trainer.step = 1
    rollout.generate([[3, 4]], 5)
    after = load_file(str(rollout.path / "adapter_model.safetensors"))
    assert any(not torch.equal(weights[key], after[key]) for key in weights)
    assert rollout.engine.removed == [1]
    assert rollout.engine.calls[-1][2].lora_int_id == 2
    params = rollout.engine.calls[0][1]
    assert all(p.max_tokens == 5 and p.temperature == 1.1 and p.top_k == -1 for p in params)
    assert params[0].seed != params[1].seed

def test_rng_replay_and_sleep_on_failure(tmp_path):
    trainer = tiny_trainer(tmp_path)
    rollout = backend(trainer)
    state = torch.get_rng_state().clone()
    rollout.generate([[3, 4]], 5)
    seed = rollout.engine.calls[-1][1][0].seed
    expected_state = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    rollout.engine.fail = True
    with pytest.raises(RuntimeError, match="generation failed"):
        rollout.generate([[3, 4]], 5)
    assert rollout.engine.asleep
    assert rollout.engine.calls[-1][1][0].seed == seed
    assert torch.equal(expected_state, torch.get_rng_state())
