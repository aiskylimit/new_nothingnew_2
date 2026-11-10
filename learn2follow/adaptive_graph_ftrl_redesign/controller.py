"""Independent actor rewards and RMS-calibrated relative FTRL feedback."""

import math

import numpy as np
import torch
from torch.distributions import Beta, kl_divergence

from .policy import LinearBeta, Standardizer


def exposure_reward(losses, grounded, gamma, coefficient):
    observations = {"loss_observations": list(losses), "grounded_observations": list(grounded)}
    if len(losses) != len(grounded) + 1 or not grounded:
        return {
            **observations,
            "actor_valid": False,
            "actor_skip_reason": "missing_loss_or_grounded_observations",
        }
    if any(x is None or not math.isfinite(x) for x in (*losses, *grounded)):
        return {
            **observations,
            "actor_valid": False,
            "actor_skip_reason": "nonfinite_loss_or_grounded_observation",
        }
    signed = [a - b for a, b in zip(losses, losses[1:])]
    positive = [max(0.0, x) for x in signed]
    discounts = [gamma**i for i in range(len(grounded))]
    lp = sum(w * x for w, x in zip(discounts, positive))
    gt = sum(w * x for w, x in zip(discounts, grounded)) / sum(discounts)
    return {
        "actor_valid": True,
        "actor_skip_reason": "",
        "loss_observations": losses,
        "signed_loss_changes": signed,
        "positive_loss_changes": positive,
        "grounded_observations": grounded,
        "G_lp": lp,
        "G_gt": gt,
        "grounded_reward_term": coefficient * gt,
        "R": lp + coefficient * gt,
    }


class HistoricalRMS:
    def __init__(self, decay, floor):
        self.decay, self.floor = decay, floor
        self.count, self.square = 0, 0.0

    @property
    def scale(self):
        return max(math.sqrt(self.square), self.floor)

    def ingest(self, value):
        self.square = (
            value * value if not self.count else self.decay * self.square + (1 - self.decay) * value * value
        )
        self.count += 1

    def diagnostics(self):
        return {**vars(self), "scale": self.scale, "floor_active": math.sqrt(self.square) <= self.floor}


class Controller:
    def __init__(self, config):
        self.cfg = config
        self.policy = LinearBeta()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.controller_lr)
        self.reward = Standardizer(
            config.normalizer_decay, config.reward_variance_floor, config.standardizer_clip
        )
        self.delta = Standardizer(
            config.normalizer_decay, config.delta_variance_floor, config.standardizer_clip
        )
        self.rms = HistoricalRMS(config.delta_rms_decay, config.delta_rms_floor)
        self.D, self.weights = 0.0, [0.5, 0.5]
        if config.teacher_selection == "hard":
            self.weights = [1.0, 0.0]
        self.actor_valid_windows, self.duel_valid_windows, self.version = 0, 0, 0
        self.last_reward, self.last_delta = 0.0, 0.0
        self.action_rng = torch.Generator().manual_seed(config.seed + 4000).get_state()

    def begin(self, index):
        # Scheduled FULL windows contribute time, never reward observations.
        total = math.ceil(self.cfg.total_updates / self.cfg.window_steps)
        state = [min(1.0, index / max(total - 1, 1)), self.last_delta, self.weights[1], self.last_reward]
        dist = self.policy.distribution(state)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.action_rng)
            alpha = self.policy.sample(state)
            self.action_rng = torch.get_rng_state()
        return {
            "state": state,
            "alpha": alpha,
            "weights_used": list(self.weights),
            "policy_version": self.version,
            "beta_a": float(dist.concentration1.detach()),
            "beta_b": float(dist.concentration0.detach()),
        }

    def update_experts(self, duel):
        result = {**duel, "ftrl_updated": False, "weights_used": list(self.weights)}
        if result["duel_valid"] and not math.isfinite(result["delta"]):
            result.update(duel_valid=False, duel_skip_reason="nonfinite_delta")
        zd = self.delta.z(result["delta"]) if result["duel_valid"] else None
        if result["duel_valid"]:
            raw = result["delta"]
            scale = self.rms.scale
            limit = self.cfg.delta_scaled_clip
            scaled = float(np.clip(raw / scale, -limit, limit))
            result.update(
                delta_raw=raw,
                delta_scaled=scaled,
                delta_scale_used=scale,
                delta_scaled_clip=limit,
                delta_floor_active=math.sqrt(self.rms.square) <= self.rms.floor,
                Z_delta=zd,
            )
            if not self.cfg.fixed_mixture:
                self.D = self.cfg.ftrl_rho * self.D + scaled
                if self.cfg.teacher_selection == "hard":
                    # Hard discounted leader; FULL wins an exact tie.
                    wm = float(self.D > 0)
                else:
                    wm = self.cfg.ftrl_min_weight + (1 - 2 * self.cfg.ftrl_min_weight) * float(
                        torch.sigmoid(torch.tensor(self.cfg.ftrl_eta * self.D, dtype=torch.float64))
                    )
                self.weights = [1 - wm, wm]
                result["ftrl_updated"] = True
            self.rms.ingest(raw)
            self.delta.ingest(raw)
            self.last_delta = zd
            self.duel_valid_windows += 1
        result.update(
            weights_next=list(self.weights),
            D=self.D,
            duel_valid_windows=self.duel_valid_windows,
            teacher_selection=self.cfg.teacher_selection,
        )
        return result

    def finish(self, pending, reward_info, duel, *, update_ftrl=True, actor_mask_weight=None):
        expert = self.update_experts(duel) if update_ftrl else duel
        result = {
            **reward_info,
            **expert,
            "weights_used": list(pending["weights_used"]),
            "controller_updated": False,
        }
        if result["actor_valid"] and not math.isfinite(result["R"]):
            result.update(actor_valid=False, actor_skip_reason="nonfinite_reward")
        zr = self.reward.z(result["R"]) if result["actor_valid"] else None
        if result["actor_valid"]:
            wm_used = pending["weights_used"][1] if actor_mask_weight is None else actor_mask_weight
            advantage = wm_used * zr
            result.update(Z_R=zr, advantage=advantage, controller_grad_norm=0.0, same_state_beta_kl=0.0)
            if actor_mask_weight is not None:
                result["actor_mask_weight_used"] = actor_mask_weight
            if self.actor_valid_windows >= self.cfg.warmup_windows and not self.cfg.frozen_controller:
                if pending["policy_version"] != self.version:
                    raise RuntimeError("Policy changed inside window")
                old = self.policy.distribution(pending["state"])
                old_detached = Beta(old.concentration1.detach().clone(), old.concentration0.detach().clone())
                self.optimizer.zero_grad(set_to_none=True)
                loss = -torch.tensor(advantage).detach() * old.log_prob(
                    torch.tensor(pending["alpha"]).detach()
                )
                loss.backward()
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.cfg.controller_max_grad_norm
                    )
                )
                if not math.isfinite(norm):
                    raise RuntimeError("Nonfinite actor gradient")
                self.optimizer.step()
                self.version += 1
                with torch.no_grad():
                    kl = float(kl_divergence(old_detached, self.policy.distribution(pending["state"])))
                result.update(controller_updated=True, controller_grad_norm=norm, same_state_beta_kl=kl)
            self.reward.ingest(result["R"])
            self.last_reward = zr
            self.actor_valid_windows += 1
        result.update(
            weights_next=list(self.weights),
            D=self.D,
            actor_valid_windows=self.actor_valid_windows,
            duel_valid_windows=self.duel_valid_windows,
            policy_version=self.version,
            normalizers={
                "reward": self.reward.diagnostics(),
                "delta": self.delta.diagnostics(),
                "delta_rms": self.rms.diagnostics(),
            },
        )
        return result

    def state_dict(self):
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "action_rng": self.action_rng,
            "reward": vars(self.reward).copy(),
            "delta": vars(self.delta).copy(),
            "rms": vars(self.rms).copy(),
            **{
                key: getattr(self, key)
                for key in (
                    "D",
                    "weights",
                    "actor_valid_windows",
                    "duel_valid_windows",
                    "version",
                    "last_reward",
                    "last_delta",
                )
            },
        }

    def restore(self, state):
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.action_rng = state["action_rng"]
        for name in ("reward", "delta", "rms"):
            getattr(self, name).__dict__.update(state[name])
        for name in (
            "D",
            "weights",
            "actor_valid_windows",
            "duel_valid_windows",
            "version",
            "last_reward",
            "last_delta",
        ):
            setattr(self, name, state[name])
