"""ASFT and VCORE ports (sgl.training.baselines). The reference formulas below follow the released code:
ASFT `EnhancedTrainer.compute_loss` (train_v2.py), VCORE `vcore_loss` (transformers-4.52.4/loss_utils.py)."""

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments

from sgl.training.baselines import anchored_loss_sum, vcore_loss, vcore_weights
from sgl.training.collator import MaskedSFTCollator
from sgl.training.losses import IGNORE_INDEX, OBJECTIVE_DFT, masked_cross_entropy
from sgl.training.train import AnchoredSFTTrainer, MaskedSFTTrainer, VCoreSFTTrainer

VOCAB = 50


def _batch(seed=0, batch=2, length=23):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, length, VOCAB, generator=generator, requires_grad=True)
    labels = torch.randint(0, VOCAB, (batch, length), generator=generator)
    labels[:, :5] = IGNORE_INDEX
    labels[1, -4:] = IGNORE_INDEX
    return logits, labels, generator


def test_asft_without_anchor_is_dft():
    logits, labels, _ = _batch()
    count = (labels[:, 1:] != IGNORE_INDEX).sum()
    ours = anchored_loss_sum(logits, torch.zeros_like(logits), labels, 0.0, chunk=7) / count
    dft = masked_cross_entropy(logits, labels, token_objective=torch.full_like(labels, OBJECTIVE_DFT))
    torch.testing.assert_close(ours, dft)


def test_asft_matches_upstream_formula_and_gradient():
    logits, labels, generator = _batch()
    ref = torch.randn(logits.shape, generator=generator)

    flat = logits[:, :-1].reshape(-1, VOCAB)
    target = labels[:, 1:].reshape(-1)
    valid = target != IGNORE_INDEX
    token_losses = torch.nn.CrossEntropyLoss(reduction="none")(flat, target)
    weights = torch.softmax(flat, -1).gather(1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1).detach()
    kl = F.kl_div(
        F.log_softmax(flat, -1), F.softmax(ref[:, :-1].reshape(-1, VOCAB), -1), reduction="none"
    ).sum(-1)
    upstream = (token_losses * weights + 0.05 * kl)[valid].sum()
    upstream_grad, = torch.autograd.grad(upstream, logits)

    ours = anchored_loss_sum(logits, ref, labels, 0.05, chunk=7)
    ours_grad, = torch.autograd.grad(ours, logits)
    torch.testing.assert_close(ours, upstream)
    torch.testing.assert_close(ours_grad, upstream_grad)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vcore_loss_matches_upstream(seed):
    logits, labels, generator = _batch(seed)
    epsilon, temperature = 1e-4, 5000.0
    previous = torch.rand(labels[:, 1:].shape, generator=generator) * 2  # CE after the probe step

    valid = labels[:, 1:] != IGNORE_INDEX
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1),
                         ignore_index=IGNORE_INDEX, reduction="none").view_as(valid)
    scores = ((ce.detach() * valid - previous * valid) / (epsilon * temperature)).masked_fill(~valid, -torch.inf)
    q = torch.softmax(scores, -1) * valid.sum(-1, keepdim=True)
    tokens = valid.sum()
    weighted = (q * ce * valid).sum() / tokens
    c = (ce * valid).sum().detach() / tokens / weighted.detach()
    upstream = weighted if c > 1 else c * weighted

    gain = (ce.detach() - previous) * valid
    ours, scale = vcore_loss(logits, labels, vcore_weights(gain, valid, epsilon, temperature), tokens)
    torch.testing.assert_close(ours, upstream)
    assert scale == pytest.approx(min(float(c), 1.0))


def _tiny():
    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
    ))


def _train(tmp_path, trainer_cls, **kwargs):
    records = [
        {"input_ids": [(7 * i + j) % VOCAB for j in range(12)], "loss_mask": [0] * 3 + [1] * 9}
        for i in range(4)
    ]
    model = _tiny()
    args = TrainingArguments(
        output_dir=str(tmp_path), per_device_train_batch_size=1, gradient_accumulation_steps=2,
        num_train_epochs=2, learning_rate=1e-2, use_cpu=True, report_to=[], save_strategy="no",
        remove_unused_columns=False, seed=0,
    )
    trainer = trainer_cls(
        model=model, args=args, train_dataset=records, data_collator=MaskedSFTCollator(pad_token_id=0), **kwargs
    )
    trainer.train()
    return trainer, torch.cat([p.detach().flatten() for p in model.parameters()])


def test_vcore_with_uniform_weights_is_plain_sft(tmp_path):
    """A huge temperature flattens the weights, so the probe must leave no trace in weights or gradients."""
    _, plain = _train(tmp_path / "a", MaskedSFTTrainer)
    trainer, vcore = _train(tmp_path / "b", VCoreSFTTrainer, probe_epsilon=1e-4, temperature=1e12)
    torch.testing.assert_close(vcore, plain, rtol=1e-5, atol=1e-6)
    stats = trainer.method_summary()["weight_stats"]
    assert stats["effective_token_fraction"] == pytest.approx(1.0)
    assert trainer.probe_batches == 2


def test_vcore_probe_reweights_tokens(tmp_path):
    _, plain = _train(tmp_path / "a", MaskedSFTTrainer)
    trainer, vcore = _train(tmp_path / "b", VCoreSFTTrainer, probe_epsilon=0.1, temperature=0.1)
    stats = trainer.method_summary()["weight_stats"]
    assert stats["nonzero_gain"] > 0.9 and stats["effective_token_fraction"] < 0.99
    assert not torch.allclose(vcore, plain)


def test_asft_trainer_keeps_the_reference_frozen(tmp_path):
    reference = _tiny()
    before = torch.cat([p.detach().flatten().clone() for p in reference.parameters()])
    _, trained = _train(tmp_path, AnchoredSFTTrainer, reference_model=reference, kl_lambda=0.05)
    torch.testing.assert_close(torch.cat([p.detach().flatten() for p in reference.parameters()]), before)
    assert not torch.allclose(trained, before)
