import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from adaptive_graph_ftrl_redesign.data import parse_example
from adaptive_graph_ftrl_redesign.config import Config
from adaptive_graph_ftrl_redesign.trainer import Trainer


class ByteTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 1, 2, 3
    all_special_ids = [0, 1, 2, 3]

    def encode(self, text, add_special_tokens=False):
        return [b + 4 for b in text.encode()]

    def decode(self, ids, skip_special_tokens=False):
        return bytes(i - 4 for i in ids if i >= 4).decode(errors="replace")

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    ):
        return (
            "User:\n" + messages[0]["content"] + "\nAssistant:\n" + ("<think>\n" if enable_thinking else "")
        )

    def get_vocab(self):
        return {str(i): i for i in range(260)}

    def save_pretrained(self, path):
        (Path(path) / "toy_tokenizer.json").write_text(json.dumps(self.get_vocab()))


def row(index=0, fragments=None, edges=None):
    fragments = fragments or ["Given x = 1.\n", "Add y = 2.\n", "Thus x+y=3. Final answer: \\boxed{3}"]
    source, offset, steps = "".join(fragments), 0, []
    for i, text in enumerate(fragments):
        steps.append({"step_id": i, "text": text, "char_start": offset, "char_end": offset + len(text)})
        offset += len(text)
    return {
        "row_index": index,
        "hypergraph": {
            "problem": f"Find x+y in problem {index}.",
            "solution": source,
            "semantic_steps": steps,
            "hyperedges": edges
            if edges is not None
            else [{"parent_steps": [0], "target_step": 1}, {"parent_steps": [1], "target_step": 2}],
        },
    }


def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    mc = Qwen3Config(
        vocab_size=260,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=2048,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=3,
        attention_dropout=0.0,
    )
    mc._attn_implementation = "eager"
    model = get_peft_model(
        Qwen3ForCausalLM(mc),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    return model


def tiny_trainer(path, rank=0, world=1, **options):
    model = tiny_model()
    settings = dict(
        output_dir=str(path),
        generation_backend="transformers",
        dtype="float32",
        lora_rank=2,
        lora_alpha=4,
        effective_batch_size=2 * world,
        gradient_accumulation_steps=2,
        control_pool_size=4,
        control_batch_size=2,
        min_valid_controls=1,
        total_updates=10,
        full_opsd_warmup_steps=2,
        window_steps=2,
        main_max_new_tokens=64,
        grounded_score_examples_per_step=1,
        grounded_reference_max_tokens=128,
        max_length=2048,
        benchmark_tasks=(),
        save_every=25,
        learning_rate=1e-3,
    )
    settings.update(options)
    trainer = Trainer(
        model,
        ByteTokenizer(),
        [parse_example(row(i), i) for i in range(12)],
        Config(**settings),
        {"sha256": "toy", "parser": "terminal_block_v1", "eligible": 12, "excluded": 0},
        rank,
        world,
    )

    def generate_many(examples, probe=False):
        assert not probe, "Redesign must not generate control reasoning"
        trainer.count("main_generation_calls", len(examples))
        tokens = trainer.tokenizer.encode("x+y=3.") + [2]
        trainer.count("main_generated_tokens", len(tokens) * len(examples))
        return [tokens for _ in examples]

    trainer.generate_many = generate_many
    return trainer
