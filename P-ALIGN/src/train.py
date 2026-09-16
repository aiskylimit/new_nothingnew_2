import os
import time

from llamafactory.model.model_utils import checkpointing as _lf_checkpointing
from llamafactory.train.tuner import run_exp
from transformers import TrainerCallback

# transformers 5.17 Trainer.train() passes every_n_layers/offload;
# LLaMA-Factory 0.9.5's patched enable() does not accept them.
_orig_gc_enable = _lf_checkpointing._gradient_checkpointing_enable


def _gradient_checkpointing_enable_compat(
    self,
    gradient_checkpointing_kwargs=None,
    use_unsloth_gc=False,
    every_n_layers=1,
    offload=False,
    **kwargs,
):
    return _orig_gc_enable(
        self,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        use_unsloth_gc=use_unsloth_gc,
    )


_lf_checkpointing._gradient_checkpointing_enable = _gradient_checkpointing_enable_compat


class StopAfterEpochCallback(TrainerCallback):
    """Stop training once `stop_epoch` epochs are done while keeping the
    scheduler horizon from num_train_epochs (cosine over 5, train 3).
    The epoch checkpoint is still saved because DefaultFlowCallback sets
    should_save before this runs and Trainer saves before checking stop."""

    def __init__(self, stop_epoch):
        self.stop_epoch = stop_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= self.stop_epoch - 1e-3:
            print(f"StopAfterEpochCallback: epoch {state.epoch:.2f} >= {self.stop_epoch}, stopping")
            control.should_training_stop = True
        return control


def _callbacks():
    stop_epoch = float(os.environ.get("PALIGN_STOP_EPOCH", "0"))
    return [StopAfterEpochCallback(stop_epoch)] if stop_epoch > 0 else []


def main():
    t0 = time.time()
    run_exp(callbacks=_callbacks())
    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"train finished in {elapsed:.1f}s ({h:02d}:{m:02d}:{s:02d})")


def _mp_fn(index):
    run_exp(callbacks=_callbacks())


if __name__ == "__main__":
    main()
