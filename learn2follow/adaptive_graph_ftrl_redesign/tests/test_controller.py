import copy
import math

import pytest
import torch

from adaptive_graph_ftrl_redesign.config import Config
from adaptive_graph_ftrl_redesign.controller import Controller, exposure_reward


def actor(reward=1.0):
    return {"actor_valid": True, "actor_skip_reason": "", "R": reward}


def duel(delta=0.001):
    return {"duel_valid": True, "duel_skip_reason": "", "delta": delta}


def test_reward_discount_signed_deltas_and_negative_log_probability():
    info = exposure_reward([0.5, 0.3, 0.4, 0.1], [-2, -3, -4], 0.95, 0.01)
    assert info["signed_loss_changes"] == pytest.approx([0.2, -0.1, 0.3])
    assert info["G_lp"] == pytest.approx(0.2 + 0.95**2 * 0.3)
    assert info["G_gt"] == pytest.approx((-2 - 3 * 0.95 - 4 * 0.95**2) / (1 + 0.95 + 0.95**2))
    assert info["R"] == pytest.approx(info["G_lp"] + 0.01 * info["G_gt"])
    assert not exposure_reward([1, float("nan")], [-1], 0.95, 0.01)["actor_valid"]
    assert not exposure_reward([1], [-1], 0.95, 0.01)["actor_valid"]


def test_independent_channels_warmup_historical_rms_no_centering():
    ctl = Controller(Config())
    assert sum(p.numel() for p in ctl.policy.parameters()) == 10
    initial = [p.detach().clone() for p in ctl.policy.parameters()]
    assert ctl.begin(1)["beta_a"] == ctl.begin(1)["beta_b"] == 2
    invalid = {"duel_valid": False, "duel_skip_reason": "missing"}
    for k in range(3):
        info = ctl.finish(ctl.begin(k + 1), actor(k + 1), invalid)
        assert not info["controller_updated"] and not info["ftrl_updated"]
        assert ctl.D == 0 and ctl.rms.count == 0
    for p, old in zip(ctl.policy.parameters(), initial):
        assert torch.equal(p, old)
    pending = ctl.begin(4)
    info = ctl.finish(pending, actor(10), duel())
    assert info["controller_updated"] and info["ftrl_updated"]
    assert info["advantage"] == pending["weights_used"][1] * info["Z_R"]
    assert info["delta_scale_used"] == 1e-4 and info["delta_scaled"] == 1
    assert ctl.weights[1] > 0.5 and ctl.D == 1
    old_reward, version = vars(ctl.reward).copy(), ctl.version
    bad_actor = {"actor_valid": False, "actor_skip_reason": "nonfinite"}
    info = ctl.finish(ctl.begin(5), bad_actor, duel())
    assert info["delta_scaled"] == 1 and ctl.D == pytest.approx(1.9)
    assert vars(ctl.reward) == old_reward and ctl.version == version
    old_d, old_count = ctl.D, ctl.rms.count
    ctl.finish(ctl.begin(6), actor(3), invalid)
    assert ctl.D == old_d and ctl.rms.count == old_count


def test_invalid_both_freeze_all_statistics_and_feedback():
    ctl = Controller(Config())
    ctl.finish(ctl.begin(1), actor(), duel())
    before = copy.deepcopy(ctl.state_dict())
    info = ctl.finish(
        ctl.begin(2),
        {"actor_valid": False, "actor_skip_reason": "bad"},
        {"duel_valid": False, "duel_skip_reason": "bad"},
    )
    after = ctl.state_dict()
    for key in ("reward", "delta", "rms", "weights", "D", "version", "last_reward", "last_delta"):
        assert before[key] == after[key]
    assert not info["controller_updated"] and not info["ftrl_updated"]


@pytest.mark.parametrize("frozen,fixed", [(True, False), (False, True)])
def test_ablations(frozen, fixed):
    ctl = Controller(Config(frozen_controller=frozen, fixed_mixture=fixed, warmup_windows=0))
    pending = ctl.begin(1)
    # Seed historical reward; a scalar never normalizes itself.
    ctl.reward.ingest(0.0)
    result = ctl.finish(pending, actor(1.0), duel())
    assert result["controller_updated"] is not frozen
    assert result["ftrl_updated"] is not fixed
    if fixed:
        assert ctl.weights == [0.5, 0.5] and ctl.D == 0
    if frozen:
        dist = ctl.policy.distribution(pending["state"])
        assert float(dist.concentration1.detach()) == float(dist.concentration0.detach()) == 2


def test_config_defaults_validation_and_actor_rng_isolation():
    cfg = Config().validate(4)
    assert cfg.seed == cfg.eval_seed == 42
    assert cfg.gradient_accumulation_steps == 8 and cfg.control_batch_size == 16
    assert cfg.full_opsd_warmup_steps == cfg.window_steps == 5
    assert cfg.gamma_discount == 0.95 and cfg.lambda_gt == 0.01
    ctl = Controller(cfg)
    rng = torch.get_rng_state().clone()
    action = ctl.begin(1)
    assert torch.equal(torch.get_rng_state(), rng)
    assert 0 < action["alpha"] < 1 and math.isfinite(action["alpha"])
    with pytest.raises(ValueError, match="floor"):
        Config(ftrl_min_weight=0.5).validate(4)
    with pytest.raises(ValueError, match="integer"):
        Config(full_opsd_warmup_steps=1.5).validate(4)


def test_hard_leader_switches_without_mixing_and_ties_choose_full():
    cfg = Config(teacher_selection="hard").validate(4)
    ctl = Controller(cfg)
    assert ctl.weights == [1.0, 0.0]
    pending = ctl.begin(1)
    first = ctl.update_experts(duel())
    assert first["weights_used"] == [1.0, 0.0]
    assert first["weights_next"] == [0.0, 1.0]
    assert first["D"] == 1.0
    second = ctl.update_experts(duel(-0.001))
    assert second["D"] == pytest.approx(-0.1)
    assert second["weights_used"] == [0.0, 1.0]
    assert second["weights_next"] == [1.0, 0.0]
    assert ctl.actor_valid_windows == ctl.reward.count == ctl.version == 0
    assert ctl.duel_valid_windows == ctl.rms.count == ctl.delta.count == 2
    before = copy.deepcopy(ctl.state_dict())
    invalid = ctl.update_experts({"duel_valid": False, "duel_skip_reason": "bad_controls"})
    assert not invalid["ftrl_updated"]
    assert ctl.D == before["D"] and ctl.weights == before["weights"]
    assert vars(ctl.rms) == before["rms"]
    ctl.D = 0
    zero = ctl.update_experts(duel(0))
    assert zero["weights_next"] == [1.0, 0.0]
    assert pending["weights_used"] == [1.0, 0.0]


def test_hard_actor_uses_realized_mask_frequency_without_double_duel_update():
    ctl = Controller(Config(teacher_selection="hard", warmup_windows=0))
    pending = ctl.begin(1)
    expert = ctl.update_experts(duel())
    ctl.reward.ingest(0.0)
    result = ctl.finish(pending, actor(1.0), expert, update_ftrl=False, actor_mask_weight=0.4)
    assert result["advantage"] == pytest.approx(0.4 * result["Z_R"])
    assert result["actor_mask_weight_used"] == 0.4
    assert ctl.duel_valid_windows == ctl.rms.count == ctl.delta.count == 1
    assert ctl.actor_valid_windows == ctl.version == 1
    assert result["weights_used"] == [1.0, 0.0]


def test_hard_ignores_soft_eta_and_floor_but_preserves_cli_default():
    from adaptive_graph_ftrl_redesign.config import load_config
    from adaptive_graph_ftrl_redesign.train import parser

    a = Controller(Config(teacher_selection="hard", ftrl_eta=1, ftrl_min_weight=0.02))
    b = Controller(Config(teacher_selection="hard", ftrl_eta=3, ftrl_min_weight=0.1))
    assert a.update_experts(duel())["weights_next"] == b.update_experts(duel())["weights_next"]
    assert Config().teacher_selection == "soft" and Config().ftrl_eta == 3
    assert parser().parse_args(["--teacher-selection", "hard"]).teacher_selection == "hard"
    cfg = load_config("adaptive_graph_ftrl_redesign/configs/graph_hard_answer.yaml").validate(4)
    assert cfg.teacher_selection == "hard" and cfg.window_steps == 5
    assert cfg.mask_teacher_include_final_answer and cfg.seed == cfg.eval_seed == 42
    with pytest.raises(ValueError, match="fixed_mixture"):
        Config(teacher_selection="hard", fixed_mixture=True).validate(4)


def test_configurable_hard_decay_rms_and_clipping():
    cfg = Config(
        teacher_selection="hard",
        ftrl_rho=0.8,
        delta_rms_decay=0.7,
        delta_rms_floor=0.002,
        delta_scaled_clip=0.5,
    ).validate(4)
    ctl = Controller(cfg)
    first = ctl.update_experts(duel(0.01))
    assert first["delta_scale_used"] == 0.002
    assert first["delta_scaled"] == first["D"] == 0.5
    second = ctl.update_experts(duel(-0.02))
    assert second["delta_scale_used"] == 0.01
    assert second["delta_scaled"] == -0.5
    assert second["D"] == pytest.approx(0.8 * 0.5 - 0.5)
    assert ctl.rms.square == pytest.approx(0.7 * 0.01**2 + 0.3 * 0.02**2)
    assert second["weights_next"] == [1.0, 0.0]
