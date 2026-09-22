"""Trainer plumbing for L = L_NLL + lambda * L_trans (mixin over MaskedSFTTrainer).

The base class owns the NLL (vanilla or SGL-masked, Eq. 9); this mixin adds the transition term
on top of *every* kept transition pair regardless of the NLL mask, scales it so lambda means
"per optimizer step" under gradient accumulation, gives the predictor its own optimizer group
and logs the diagnostics the method spec asks for.
"""

import json
import os
from collections import defaultdict
from contextlib import contextmanager

import torch
from transformers import TrainerCallback

from transition_loss import HiddenStateCapture, TransitionPredictor, find_decoder_layers, transition_loss

PREDICTOR_FILE = "trans_predictor.pt"
PREDICTOR_ATTR = "trans_predictor"
TRANS_INPUT_KEYS = ("step_id", "step_end", "pair_src", "num_steps")

# Entry points into DeepSpeed's backward epilogue (the ZeRO gradient reduction), muted around the
# grad-norm probe on whichever of the engine/optimizer owns them; see _backward_hooks_muted.
BACKWARD_EPILOGUE_METHODS = ("_backward_epilogue", "run_grad_acc_post_hooks")


def _noop(*args, **kwargs):
    return None


def _scalar_state(obj):
    """The object's plain-scalar attributes -- DeepSpeed keeps its backward state machine in these."""
    return {k: v for k, v in obj.__dict__.items() if v is None or isinstance(v, (bool, int, float, str))}


def attach_predictor(model, hidden: int) -> TransitionPredictor:
    """Register the predictor as a submodule of the (PEFT) model.

    A submodule, not a free-standing module: DeepSpeed/Trainer only move, checkpoint and
    optimize parameters that live under the model they were handed. PEFT's adapter save ignores
    it (not an adapter weight), so adapter checkpoints stay loadable by vLLM as before; the
    predictor is written next to them as trans_predictor.pt by PredictorCheckpointCallback.
    """
    d_model = model.config.hidden_size
    predictor = TransitionPredictor(d_model, hidden).float()
    predictor.to(next(model.parameters()).device)
    setattr(model, PREDICTOR_ATTR, predictor)
    return predictor


def predictor_of(model) -> TransitionPredictor:
    return getattr(model, PREDICTOR_ATTR)


class PredictorCheckpointCallback(TrainerCallback):
    """Save the predictor into every checkpoint-N/ (rank 0), so --resume can pick it up."""

    def __init__(self, predictor: torch.nn.Module):
        self.predictor = predictor

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}", PREDICTOR_FILE)
            torch.save(self.predictor.state_dict(), path)


class TransitionLossMixin:
    """Mix in *before* MaskedSFTTrainer: `class TransitionSFTTrainer(TransitionLossMixin, MaskedSFTTrainer)`.

    Constructor kwargs (all popped before reaching Trainer):
        trans_lambda: weight of L_trans at full strength.
        trans_layer: 0-based decoder layer whose output is used for source and target.
        trans_lr: learning rate of the predictor's param group (no weight decay).
        trans_lambda_warmup: ramp lambda linearly from 0 over the LR warmup steps.
        trans_shuffle_targets: the shuffled-target control.
        trans_grad_log_interval: every N optimizer steps, measure ||grad L_NLL|| and
            lambda*||grad L_trans|| on the LoRA parameters (two extra backward passes on that
            one microbatch; 0 disables).
    """

    def __init__(self, *args, **kwargs):
        self.trans_lambda = float(kwargs.pop("trans_lambda"))
        self.trans_layer = int(kwargs.pop("trans_layer"))
        self.trans_lr = float(kwargs.pop("trans_lr"))
        self.trans_lambda_warmup = bool(kwargs.pop("trans_lambda_warmup", True))
        self.trans_shuffle_targets = bool(kwargs.pop("trans_shuffle_targets", False))
        self.trans_grad_log_interval = int(kwargs.pop("trans_grad_log_interval", 50))
        super().__init__(*args, **kwargs)
        # Registered on the raw layer object before DeepSpeed/DDP wrap the model; wrappers
        # keep the same module instances, so the hook keeps firing.
        layers = find_decoder_layers(self.model)
        if not 0 <= self.trans_layer < len(layers):
            raise ValueError(f"--trans-layer {self.trans_layer} out of range for {len(layers)} layers")
        self._capture = HiddenStateCapture(layers[self.trans_layer])
        self._predictor = predictor_of(self.model)
        self._lora_params = [
            p for n, p in self.model.named_parameters() if p.requires_grad and PREDICTOR_ATTR not in n
        ]
        self._metric_sums = defaultdict(float)
        self._metric_counts = defaultdict(int)
        self._grad_logged_step = -1
        self.add_callback(PredictorCheckpointCallback(self._predictor))

    # ---- optimizer: predictor in its own group (own LR, no weight decay) ----
    def create_optimizer(self, model=None):
        opt_model = self.model if model is None else model
        if self.optimizer is None:
            decay_names = set(self.get_decay_parameter_names(opt_model))
            groups = {"decay": [], "no_decay": [], "predictor": []}
            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                if PREDICTOR_ATTR in name:
                    groups["predictor"].append(param)
                elif name in decay_names:
                    groups["decay"].append(param)
                else:
                    groups["no_decay"].append(param)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            param_groups = [
                {"params": groups["decay"], "weight_decay": self.args.weight_decay},
                {"params": groups["no_decay"], "weight_decay": 0.0},
                {"params": groups["predictor"], "weight_decay": 0.0, "lr": self.trans_lr},
            ]
            self.optimizer = optimizer_cls([g for g in param_groups if g["params"]], **optimizer_kwargs)
        return self.optimizer

    # ---- lambda schedule ----
    def current_lambda(self) -> float:
        if not self.trans_lambda_warmup:
            return self.trans_lambda
        warmup = self.args.get_warmup_steps(self.state.max_steps) if self.state.max_steps else 0
        if warmup <= 0:
            return self.trans_lambda
        return self.trans_lambda * min(1.0, self.state.global_step / warmup)

    # ---- loss ----
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        trans_inputs = {key: inputs.pop(key) for key in TRANS_INPUT_KEYS if key in inputs}
        self._capture.arm()
        loss_nll, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs
        )
        hidden = self._capture.take()  # [B, T, d]

        losses, stats_list = [], []
        for b in range(hidden.size(0)):
            step_end = trans_inputs["step_end"][b]
            pair_src = trans_inputs["pair_src"][b]
            loss_b, stats_b = transition_loss(
                hidden[b],
                trans_inputs["step_id"][b],
                step_end[step_end >= 0],
                pair_src[pair_src >= 0],
                self._predictor,
                shuffle_targets=self.trans_shuffle_targets,
            )
            losses.append(loss_b)
            stats_list.append(stats_b)
        loss_trans = torch.stack(losses).mean()

        # The NLL is already sum/Z over the whole optimizer step (model_accepts_loss_kwargs), so
        # Trainer does not divide by the accumulation count; L_trans is a per-microbatch mean and
        # must be, or lambda would silently scale with --gradient-accumulation-steps.
        lam = self.current_lambda()
        accumulation = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
        loss_trans_scaled = lam * loss_trans / accumulation
        loss = loss_nll + loss_trans_scaled

        if self.model.training:
            # loss_nll here is one microbatch's share of the step-level sum/Z; scaling by the
            # accumulation count puts it on the same per-step scale as loss_trans and `loss`.
            self._record(loss_nll * accumulation, loss_trans, stats_list, lam)
            if self._should_log_grads():
                self._log_grad_norms(loss_nll, loss_trans_scaled)
        return (loss, outputs) if return_outputs else loss

    # ---- diagnostics ----
    def _record(self, loss_nll, loss_trans, stats_list, lam):
        scored = [s for s in stats_list if s.pairs > 0]
        values = {
            "loss_nll": float(loss_nll.detach()),
            "loss_trans": float(loss_trans.detach()),
            "trans_lambda": lam,
            "trans_pairs": float(sum(s.pairs for s in stats_list)) / max(1, len(stats_list)),
        }
        if scored:
            for key in ("cos", "copy_cos", "raw_step_cos", "ztilde_norm"):
                values[f"trans_{key}"] = sum(getattr(s, key) for s in scored) / len(scored)
        for key, value in values.items():
            self._metric_sums[key] += value
            self._metric_counts[key] += 1

    def _should_log_grads(self) -> bool:
        interval = self.trans_grad_log_interval
        step = self.state.global_step
        if interval <= 0 or step % interval != 0 or step == self._grad_logged_step:
            return False
        self._grad_logged_step = step
        return True

    @contextmanager
    def _backward_hooks_muted(self):
        """Hide the diagnostic backward passes from DeepSpeed.

        DeepSpeedEngine.forward() hooks its own output, so *any* backward reaching the loss runs
        the engine's backward prologue and queues its epilogue -- the ZeRO reduction -- even when
        the backward is torch.autograd.grad(loss, params), which returns the gradients instead of
        accumulating them into .grad. The per-parameter reduce hooks sit on grad accumulation and
        so never fire for the probe, leaving the ipg buckets empty for the epilogue to reduce:
        "IndexError: list index out of range" in reduce_ipg_grads.

        Muting the epilogue is only half of it. The prologue leaves the engine mid-backward
        (backward_active_depth) with its post-backward callback marked as queued, which would make
        Trainer's real backward skip queueing its own, so the scalars holding that state machine
        are rolled back too: the probe has to look like it never happened. A no-op without
        DeepSpeed, where nothing hooks the graph.
        """
        engine = next(
            (obj for obj in (getattr(self, "deepspeed", None), self.model_wrapped)
             if hasattr(obj, "_backward_epilogue")),
            None,
        )
        if engine is None:  # DDP or single GPU: nothing hooks the graph, nothing to undo
            yield
            return
        objects = [obj for obj in (engine, getattr(engine, "optimizer", None)) if obj is not None]
        saved = [(obj, _scalar_state(obj)) for obj in objects]
        muted = [(obj, name) for obj in objects for name in BACKWARD_EPILOGUE_METHODS if hasattr(obj, name)]

        for obj, name in muted:
            setattr(obj, name, _noop)
        try:
            yield
        finally:
            for obj, name in muted:
                obj.__dict__.pop(name, None)  # back to the class method
            for obj, state in saved:
                obj.__dict__.update(state)

    def _log_grad_norms(self, loss_nll, loss_trans_scaled):
        """||grad L_NLL|| vs lambda*||grad L_trans|| on the LoRA parameters, one microbatch.

        Plain autograd on the live graph (retain_graph so Trainer's real backward still runs);
        it never touches .grad, so the per-parameter reduction hooks (which sit on grad
        accumulation) stay out of it, and _backward_hooks_muted holds off the engine-level ones.
        With max_grad_norm=1 and batch 1, a ratio persistently above ~0.3-0.5 means the auxiliary
        term is eating the NLL update: lower lambda.
        """
        params = [p for p in self._lora_params if p.requires_grad]

        def norm_of(loss):
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            squares = [(g.float() ** 2).sum() for g in grads if g is not None]
            return float(torch.stack(squares).sum().sqrt()) if squares else 0.0

        try:
            with self._backward_hooks_muted():
                g_nll = norm_of(loss_nll)
                g_trans = norm_of(loss_trans_scaled) if loss_trans_scaled.requires_grad else 0.0
        except Exception as exc:  # a diagnostic is never worth taking the run down for
            self.trans_grad_log_interval = 0
            if self.is_world_process_zero():
                print(f"grad-norm diagnostic failed ({type(exc).__name__}: {exc}); disabled")
            return
        self._metric_sums["grad_nll_lora"] += g_nll
        self._metric_counts["grad_nll_lora"] += 1
        self._metric_sums["grad_trans_lora"] += g_trans
        self._metric_counts["grad_trans_lora"] += 1
        self._metric_sums["grad_trans_ratio"] += g_trans / max(g_nll, 1e-12)
        self._metric_counts["grad_trans_ratio"] += 1

    def log(self, logs, *args, **kwargs):
        # Rank-local means over the microbatches since the previous log line (DeepSpeed prints
        # rank 0 only); merged into the same dict as loss/grad_norm/lr so they land in log_history.
        for key, total in self._metric_sums.items():
            logs[key] = round(total / max(1, self._metric_counts[key]), 6)
        self._metric_sums.clear()
        self._metric_counts.clear()
        super().log(logs, *args, **kwargs)

    # ---- persistence ----
    def save_predictor(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self._predictor.state_dict(), os.path.join(output_dir, PREDICTOR_FILE))
        (json.dump(
            {
                "trans_lambda": self.trans_lambda,
                "trans_layer": self.trans_layer,
                "trans_lr": self.trans_lr,
                "trans_lambda_warmup": self.trans_lambda_warmup,
                "trans_shuffle_targets": self.trans_shuffle_targets,
                "predictor_hidden": self._predictor.up.out_features,
                "d_model": self._predictor.up.in_features,
            },
            open(os.path.join(output_dir, "trans-config.json"), "w"),
            indent=2,
        ))

    def load_predictor(self, checkpoint_dir: str) -> bool:
        path = os.path.join(checkpoint_dir, PREDICTOR_FILE)
        if not os.path.isfile(path):
            return False
        state = torch.load(path, map_location=next(self._predictor.parameters()).device)
        self._predictor.load_state_dict(state)
        return True
