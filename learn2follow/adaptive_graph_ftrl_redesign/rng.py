"""Preserve random streams across virtual updates, rollout and resume."""

import random
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda_current"] = torch.cuda.get_rng_state(torch.cuda.current_device()).tolist()
    return state

def restore_rng(state):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value

    random.setstate(tuples(state["python"]))
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if "cuda_current" in state:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG cannot be restored on CPU")
        torch.cuda.set_rng_state(
            torch.tensor(state["cuda_current"], dtype=torch.uint8), torch.cuda.current_device()
        )
