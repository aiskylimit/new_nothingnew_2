import copy
from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

from adaptive_graph_ftrl_redesign.checkpoint import restore_checkpoint, save_checkpoint
from adaptive_graph_ftrl_redesign.trainer import canonical_answer
from .fixtures import tiny_trainer


def test_full_warmup_transition_and_no_control_generation(tmp_path):
    trainer = tiny_trainer(tmp_path)
    ctl = copy.deepcopy(trainer.control_sampler.state_dict())
    grounded_rng = copy.deepcopy(trainer.grounded_rng.bit_generator.state)
    first, second = trainer.run_update(), trainer.run_update()
    for r in (first, second):
        assert r["phase"] == "full" and r["alpha"] is None and r["weights_used"] == [1, 0]
        assert r["latent_weights"] == [0.5, 0.5]
        assert r["counts"].get("mask_teacher_forward_calls", 0) == 0
        assert r["counts"].get("grounded_teacher_forward_calls", 0) == 0
        assert r["mask_token_percent"] == r["mask_step_percent"] == 0
    assert trainer.controller.actor_valid_windows == trainer.controller.duel_valid_windows == 0
    assert trainer.controller.D == 0 and trainer.controller.weights == [0.5, 0.5]
    assert trainer.control_sampler.state_dict() == ctl
    assert trainer.grounded_rng.bit_generator.state == grounded_rng
    step = float(next(iter(trainer.optimizer.state.values()))["step"])
    third = trainer.run_update()
    assert third["phase"] == "adaptive" and third["weights_used"] == [0.5, 0.5]
    assert third["counts"]["main_generation_calls"] == 2
    assert third["counts"]["mask_teacher_forward_calls"] == 2
    assert third["counts"]["grounded_teacher_forward_calls"] == 1
    assert third["counts"]["control_generation_calls"] == 0
    assert float(next(iter(trainer.optimizer.state.values()))["step"]) == step + 1
    assert trainer.scheduler.last_epoch == 3
    last = trainer.run_update()
    f = last["window_feedback"]
    assert f["duel_valid"] and f["actor_valid"]
    assert f["ftrl_updated"] and not f["controller_updated"]
    assert len(f["loss_observations"]) == 3 and len(f["grounded_observations"]) == 2
    assert last["counts"]["pure_expert_gradient_calls"] == 4
    assert last["counts"]["main_student_forward_calls"] == 2
    assert last["counts"]["replay_student_forward_calls"] == 2
    assert last["counts"]["control_forward_calls"] == 6
    assert not list(tmp_path.glob("**/graph-replay-*"))


def test_question_only_control_and_causal_answer_tokens(tmp_path):
    trainer = tiny_trainer(tmp_path)
    ex = trainer.examples[trainer.control_ids[0]]
    record = trainer.control_inputs(ex)
    text = trainer.tokenizer.decode(record["prefix"])
    assert ex.problem in text and trainer.config.direct_instruction in text
    assert ex.reasoning_source not in text and "Given x" not in text
    assert text.endswith("Final answer: \\boxed{")
    assert trainer.tokenizer.decode(record["labels"]) == "3"
    trainer.model.eval()
    with torch.no_grad():
        actual = trainer.answer_loss(record)
        ids = torch.tensor([record["prefix"] + record["labels"]])
        logits = trainer.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        start = len(record["prefix"]) - 1
        expected = F.cross_entropy(
            logits[0, start : start + len(record["labels"])], torch.tensor(record["labels"])
        )
    assert torch.allclose(actual, expected)
    assert canonical_answer(r"Final answer: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert canonical_answer("Final answer: 7") == "7"


def test_grounded_score_shift_fixed_mask_independent_span_and_answer_flag(tmp_path):
    trainer = tiny_trainer(tmp_path, grounded_reference_max_tokens=35)
    ex = next(iter(trainer.examples.values()))
    span = trainer.grounded_labels(ex)
    labels = span["labels"]
    assert span["reference_truncated"] and len(labels) == 35
    assert trainer.tokenizer.decode(labels).endswith(ex.final_answer_block)
    kwargs = trainer.logits_kwargs(ex, labels, "retained context")
    prefix = trainer.prompt(ex, "retained context")
    assert kwargs["input_ids"][0].tolist() == prefix + labels
    score, diagnostics = trainer.grounded_score(ex, "retained context")
    with torch.no_grad():
        ids = torch.tensor([prefix + labels])
        z = trainer.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
        expected = -F.cross_entropy(
            z[len(prefix) - 1 : len(prefix) - 1 + len(labels)].float()
            / trainer.config.distillation_temperature,
            torch.tensor(labels),
        )
    assert score == pytest.approx(float(expected), abs=1e-6)
    trainer.config.mask_teacher_include_final_answer = False
    assert trainer.grounded_labels(ex)["labels"] == labels
    assert diagnostics["reference_scored_tokens"] == 35
    assert score < 0


def test_full_baseline_bypasses_all_adaptivity_and_short_window(tmp_path):
    trainer = tiny_trainer(tmp_path, variant="full", total_updates=3)
    rows = [trainer.run_update() for _ in range(3)]
    assert all(r["weights_used"] == [1, 0] and r["alpha"] is None for r in rows)
    windows = [json.loads(x) for x in trainer.windows.path.read_text().splitlines()]
    assert [w["length"] for w in windows] == [2, 1]
    assert trainer.controller.D == 0 and trainer.control_sampler.cursor == 0


@pytest.mark.parametrize("bad_channel", ["actor", "duel"])
def test_invalid_channels_are_independent_and_training_continues(tmp_path, bad_channel):
    trainer = tiny_trainer(tmp_path)
    if bad_channel == "actor":
        trainer.grounded_score = lambda *args: (None, {})
    else:
        trainer.control_inputs = lambda ex: {"qid": ex.qid, "valid": False, "reason": "empty_gold"}
    for _ in range(4):
        result = trainer.run_update()
    feedback = result["window_feedback"]
    assert feedback["actor_valid"] is (bad_channel != "actor")
    assert feedback["duel_valid"] is (bad_channel != "duel")
    assert trainer.controller.actor_valid_windows == int(bad_channel != "actor")
    assert trainer.controller.duel_valid_windows == int(bad_channel != "duel")
    assert trainer.step == trainer.scheduler.last_epoch == 4


@pytest.mark.parametrize("teacher_selection", ["soft", "hard"])
@pytest.mark.parametrize("checkpoint_step", [1, 3])
def test_exact_resume_warmup_and_pending_adaptive_window(tmp_path, checkpoint_step, teacher_selection):
    original = tiny_trainer(tmp_path / "original", teacher_selection=teacher_selection)
    for _ in range(checkpoint_step):
        original.run_update()
    pending = copy.deepcopy(original.pending)
    save_checkpoint(original)
    expected = [original.run_update() for _ in range(10 - checkpoint_step)]
    weights = {n: p.clone() for n, p in original.params.items()}
    assert original.controller.actor_valid_windows == 4
    assert original.controller.duel_valid_windows == (8 if teacher_selection == "hard" else 4)
    assert original.controller.version == 1
    resumed = tiny_trainer(tmp_path / "resumed", teacher_selection=teacher_selection)
    restore_checkpoint(resumed, tmp_path / "original" / f"checkpoint-{checkpoint_step}")
    assert resumed.pending == pending
    actual = [resumed.run_update() for _ in range(10 - checkpoint_step)]
    for a, b in zip(actual, expected):
        for key in (
            "loss",
            "weights_used",
            "alpha",
            "grounded_score",
            "phase",
            "window_feedback",
            "expert_feedback",
        ):
            assert a.get(key) == b.get(key)
    for n, p in resumed.params.items():
        assert torch.equal(p, weights[n])
    assert resumed.controller.version == 1
    bad = tiny_trainer(tmp_path / "bad", lambda_gt=0.0, teacher_selection=teacher_selection)
    with pytest.raises(ValueError, match="lambda_gt"):
        restore_checkpoint(bad, tmp_path / "original" / f"checkpoint-{checkpoint_step}")


@pytest.mark.parametrize("warmup_steps", [0, 2])
def test_hard_teacher_uses_pure_target_and_gradient_every_step(tmp_path, monkeypatch, warmup_steps):
    trainer = tiny_trainer(
        tmp_path,
        teacher_selection="hard",
        total_updates=warmup_steps + 4,
        full_opsd_warmup_steps=warmup_steps,
        max_grad_norm=0,
    )
    for _ in range(warmup_steps):
        trainer.run_update()
    original = trainer.expert_gradients
    expected_gradient = {n: torch.zeros_like(p) for n, p in trainer.params.items()}
    snapshots = []

    def expert_gradients(ex, tokens, full, mask, targets):
        index = 0 if trainer.controller.weights[0] == 1 else 1
        for chunk, q in zip(trainer.chunks(len(tokens)), targets):
            teacher = (full, mask)[index]
            assert torch.equal(q, trainer.probabilities(teacher[chunk]).cpu())
        value, gradients = original(ex, tokens, full, mask, targets)
        for name, g in gradients[index].items():
            expected_gradient[name].add_(g / trainer.config.effective_batch_size)
        return value, gradients

    def check_commit(optimizer, args, kwargs):
        for name, p in trainer.params.items():
            assert torch.equal(p.grad, expected_gradient[name])
            expected_gradient[name].zero_()

    def virtual(*args):
        snapshots.append(trainer.step)
        return {
            "duel_valid": True,
            "duel_skip_reason": "",
            "delta": 0.001 if trainer.step % 2 == 0 else -0.001,
        }

    trainer.expert_gradients = expert_gradients
    trainer.optimizer.register_step_pre_hook(check_commit)
    monkeypatch.setattr("adaptive_graph_ftrl_redesign.trainer.virtual_duel", virtual)
    original_update = trainer.controller.update_experts

    def update_after_commit(duel):
        assert snapshots[-1] + 1 == trainer.step
        assert float(next(iter(trainer.optimizer.state.values()))["step"]) == trainer.step
        return original_update(duel)

    trainer.controller.update_experts = update_after_commit
    rows = [trainer.run_update() for _ in range(4)]
    assert [r["weights_used"] for r in rows] == [[1, 0], [0, 1], [1, 0], [0, 1]]
    assert [r["teacher_used"] for r in rows] == ["FULL", "MASK", "FULL", "MASK"]
    assert rows[0]["alpha"] == rows[1]["alpha"] and rows[2]["alpha"] == rows[3]["alpha"]
    assert trainer.controller.duel_valid_windows == 4 and trainer.controller.actor_valid_windows == 2
    assert all(r["expert_feedback"]["ftrl_updated"] for r in rows)
    assert all(r["counts"]["pure_expert_gradient_calls"] == 4 for r in rows)
    assert "replay_student_forward_calls" not in rows[0]["counts"]
    assert rows[1]["counts"]["replay_student_forward_calls"] == 2
    assert rows[1]["window_feedback"]["actor_mask_weight_used"] == 0.5
    assert snapshots == list(range(warmup_steps, warmup_steps + 4))


def test_pre_clip_config_checkpoint_preserves_original_behavior(tmp_path):
    original = tiny_trainer(tmp_path / "original", teacher_selection="hard")
    original.run_update()
    save_checkpoint(original)
    checkpoint = tmp_path / "original" / "checkpoint-1"
    state_path = checkpoint / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    del state["config"]["delta_scaled_clip"]
    torch.save(state, state_path)
    resumed = tiny_trainer(tmp_path / "resumed", teacher_selection="hard")
    restore_checkpoint(resumed, checkpoint)
    assert resumed.step == original.step and resumed.config.delta_scaled_clip == 1.0
    incompatible = tiny_trainer(tmp_path / "different", teacher_selection="hard", delta_scaled_clip=0.5)
    with pytest.raises(ValueError, match="delta_scaled_clip"):
        restore_checkpoint(incompatible, checkpoint)


def test_hard_invalid_controls_hold_teacher_without_decay(tmp_path):
    trainer = tiny_trainer(tmp_path, teacher_selection="hard")
    trainer.control_inputs = lambda ex: {"qid": ex.qid, "valid": False, "reason": "bad_gold"}
    rows = [trainer.run_update() for _ in range(4)]
    for row in rows[2:]:
        assert row["weights_used"] == [1, 0]
        assert not row["expert_feedback"]["ftrl_updated"]
        assert row["expert_feedback"]["duel_skip_reason"] == "insufficient_valid_controls"
    assert trainer.controller.D == 0 and trainer.controller.rms.count == 0
    assert trainer.controller.actor_valid_windows == 1


def test_resume_rejects_different_teacher_selection_and_accepts_old_soft_config(tmp_path):
    trainer = tiny_trainer(tmp_path / "original")
    trainer.run_update()
    save_checkpoint(trainer)
    path = tmp_path / "original/checkpoint-1"
    hard = tiny_trainer(tmp_path / "hard", teacher_selection="hard")
    with pytest.raises(ValueError, match="teacher_selection"):
        restore_checkpoint(hard, path)
    state = torch.load(path / "training_state.pt", weights_only=False)
    state["config"].pop("teacher_selection")
    torch.save(state, path / "training_state.pt")
    resumed = tiny_trainer(tmp_path / "soft")
    restore_checkpoint(resumed, path)
    assert resumed.step == 1 and resumed.controller.weights == [0.5, 0.5]


def test_all_alias_is_not_a_graph_observation(tmp_path):
    trainer = tiny_trainer(tmp_path)
    original_prompt = trainer.prompt
    trainer.prompt = lambda ex, reference=None: original_prompt(
        ex, ex.reference_solution if reference is not None else None
    )
    for _ in range(4):
        result = trainer.run_update()
    f = result["window_feedback"]
    assert not f["actor_valid"] and not f["duel_valid"]
    assert f["actor_skip_reason"] == "all_window_masks_alias_full"
    assert f["duel_skip_reason"] == "all_duel_masks_alias_full"


def test_pure_experts_combine_to_actual_gradient_without_grad_leaks(tmp_path):
    trainer = tiny_trainer(tmp_path)
    ex = trainer.examples[trainer.main_ids[0]]
    tokens = trainer.generate_many([ex])[0]
    full = trainer.score(ex, tokens, ex.reference_solution, "full_teacher")
    mask = trainer.score(ex, tokens, ex.final_answer_block, "mask_teacher")
    weights = [0.35, 0.65]
    targets = [
        (weights[0] * trainer.probabilities(full[c]) + weights[1] * trainer.probabilities(mask[c])).cpu()
        for c in trainer.chunks(len(tokens))
    ]
    old = {}
    for n, p in trainer.params.items():
        p.grad = torch.ones_like(p)
        old[n] = p.grad.clone()
    value, gradients = trainer.expert_gradients(ex, tokens, full, mask, targets)
    for n, p in trainer.params.items():
        assert torch.equal(p.grad, old[n])
    trainer.optimizer.zero_grad(set_to_none=True)
    actual_value = trainer.main_backward(ex, tokens, targets, 1)
    assert value == pytest.approx(actual_value, abs=1e-6)
    for n, p in trainer.params.items():
        expected = weights[0] * gradients[0][n] + weights[1] * gradients[1][n]
        assert torch.allclose(p.grad, expected, atol=1e-8, rtol=1e-5)
    assert all(p.grad is None for p in trainer.model.parameters() if not p.requires_grad)


def test_invalid_gold_and_reference_capacity_are_reported(tmp_path):
    trainer = tiny_trainer(tmp_path)
    ex = next(iter(trainer.examples.values()))
    record = trainer.control_inputs(replace(ex, gold_answer=r"\boxed{unfinished"))
    assert not record["valid"] and record["reason"] == "gold_tokenization_error"
    record = trainer.control_inputs(replace(ex, gold_answer=""))
    assert not record["valid"] and record["reason"] == "empty_gold_label_or_template"
    trainer.config.grounded_reference_max_tokens = 2
    score, diagnostics = trainer.grounded_score(ex, ex.final_answer_block)
    assert score is None and diagnostics["reference_scored_tokens"] == 0


def test_progress_and_step_metrics_follow_se_style(tmp_path, capsys):
    trainer = tiny_trainer(tmp_path, total_updates=4)
    trainer.train()
    output = capsys.readouterr()
    assert "100%" in output.err
    assert "'mask_token_percent':" in output.out and "'mask_step_percent':" in output.out
    assert "'phase': 'full'" in output.out and "'phase': 'adaptive'" in output.out
    assert "'controls': '2/2'" in output.out and "'window': 1" in output.out
    assert "generation start" not in output.out and "generation done" not in output.out
    rows = [json.loads(x) for x in trainer.metrics.path.read_text().splitlines()]
    assert len(rows) == 4 and [r["step"] for r in rows] == [1, 2, 3, 4]
    assert (tmp_path / "checkpoint-4" / "adapter_model.safetensors").is_file()
