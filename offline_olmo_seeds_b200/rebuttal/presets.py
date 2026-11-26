"""Single source of truth for the multi-seed rebuttal runs.

Every method (TROPIC, OPSD, RLSD, SDPO) is trained by the SAME loop
(rebuttal/train.py) with the SAME budget, sampling, optimizer, checkpoint
grid and evaluation protocol. Only the loss and the teacher context differ.
Change a value here and it changes for all four methods at once.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Seeds and checkpoint grid
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2, 3]

# One grid for EVERY method. The paper evaluated TROPIC at 10/15/20/25/40 but
# OPSD only at 25/50/75/100; a shared grid removes that asymmetry.
CHECKPOINT_STEPS = [10, 15, 20, 25, 40, 50, 75, 100]

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# Reported benchmarks (as in the paper). Every saved checkpoint of every run
# is evaluated on all of them.
TEST_BENCHMARKS = ["aime25", "aime26", "hmmt25"]

EVAL_K = 12               # samples per problem (avg@12 / pass@12)
EVAL_NUM_PROBLEMS = 30
EVAL_MAX_NEW_TOKENS = 38912

# ---------------------------------------------------------------------------
# Shared training recipe (paper Table 10)
# ---------------------------------------------------------------------------

TRAIN = dict(
    training_steps=100,
    rollouts_per_step=32,      # SAME number of generated rollouts per step for all methods
    max_new_tokens=1024,
    lr=5e-6,
    weight_decay=0.01,
    max_grad_norm=0.1,
    max_length=20000,          # token cap on rendered teacher/student prompts (OPSD's --max_length)
    # Rollout sampling. OPSD's official recipe (and the setting the paper's
    # Qwen3-8B TROPIC/OPSD runs actually used) is 1.1 / 0.95 / 20. Paper
    # Table 10 says 1.0 / 1.0 / -1, which is what the old RLSD/SDPO/Olmo
    # offline runs used. Pick ONE for all methods and make Table 10 match.
    rollout_temperature=1.1,
    rollout_top_p=0.95,
    rollout_top_k=20,
)

LORA = dict(
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "qwen3-8b": dict(
        hf_name="Qwen/Qwen3-8B",
        empty_think_suffix=None,          # Qwen3 has a real enable_thinking switch
        eval_temperature=1.0, eval_top_p=0.95, eval_top_k=-1,
    ),
    "olmo3-7b-think": dict(
        hf_name="allenai/Olmo-3-7B-Think",
        # No enable_thinking branch in OLMo-3's template: close the auto-opened
        # <think> tag for the (non-thinking) student prompt.
        empty_think_suffix="\n\n</think>\n\n",
        # Olmo-3's recommended decoding (used by the paper's Olmo runs).
        eval_temperature=0.6, eval_top_p=0.95, eval_top_k=-1,
    ),
    "qwen3-4b": dict(
        hf_name="Qwen/Qwen3-4B",
        empty_think_suffix=None,
        eval_temperature=1.0, eval_top_p=0.95, eval_top_k=-1,
    ),
    # CPU smoke test only (scripts never use it for real results).
    "smoke-qwen2.5-0.5b": dict(
        hf_name="Qwen/Qwen2.5-0.5B-Instruct",
        empty_think_suffix=None,
        eval_temperature=1.0, eval_top_p=0.95, eval_top_k=-1,
        no_thinking_kwarg=True,
    ),
}

# ---------------------------------------------------------------------------
# Methods. Values can be overridden per run with  train.py --set key=value
# (e.g. --set epsilon=0.3 --variant eps0.3 for the sensitivity sweep).
# ---------------------------------------------------------------------------

METHOD_DEFAULTS = {
    "tropic": dict(epsilon=0.1, beta=0.1, alpha=0.25, top_k=64, default_mass=1e-5),
    # OPSD's official 8B recipe: forward KL (beta=0), full vocabulary,
    # per-token clip 0.06, temperature 1.1 inside the loss, frozen teacher.
    "opsd": dict(beta=0.0, jsd_token_clip=0.06, temperature=1.1),
    # RLSD (Yang et al.): group-relative advantage reweighted by the
    # answer-conditioned teacher; teacher snapshot refreshed every 20 steps
    # (reference launch script), lambda 0.5 -> 0 over 50 steps.
    "rlsd": dict(group_size=8, epsilon_w=0.2, lambda_init=0.5, lambda_decay_steps=50, teacher_sync_interval=20),
    # SDPO (Huebotter et al.): live self-teacher that sees a successful
    # rollout from the same group; JSD (alpha=0.5) over student top-100.
    "sdpo": dict(group_size=8, top_k=100, default_mass=1e-5, alpha=0.5),
}

# Frozen-initial teacher (OPSD recipe) for TROPIC/OPSD; RLSD/SDPO use the
# live / periodically synced student weights as teacher.
FIXED_TEACHER = {"tropic": True, "opsd": True, "rlsd": False, "sdpo": False}

METHOD_LABELS = {"tropic": "TROPIC", "opsd": "OPSD", "rlsd": "RLSD", "sdpo": "SDPO"}
METHOD_ORDER = ["tropic", "opsd", "rlsd", "sdpo"]


def method_dir_name(method: str, variant: str | None = None) -> str:
    return method if not variant else f"{method}-{variant}"


def run_dir(runs_root: str | Path, model: str, method: str, seed: int, variant: str | None = None) -> Path:
    """runs/<model>/<method>[-<variant>]/seed<k>/  (ckpt/, eval/, train_log.jsonl)."""
    return Path(runs_root) / model / method_dir_name(method, variant) / f"seed{seed}"


def base_run_dir(runs_root: str | Path, model: str) -> Path:
    """Evaluation of the untrained base model (step 0), shared by all methods."""
    return Path(runs_root) / model / "base" / "seed0"
