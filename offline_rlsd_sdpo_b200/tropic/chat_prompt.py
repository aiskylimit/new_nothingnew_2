"""Minimal chat-template rendering helper shared by the RLSD/SDPO baselines'
own prompt builders (tropic/rlsd.py, tropic/sdpo.py) and their data loader
(tropic/verifier_data.py).

Deliberately NOT importing tropic.data's private `_apply_chat_template_ids`
(kept these baseline files dependency-free of any *internal* helper of the
existing TROPIC-P/OPSD code, per an explicit "don't touch/lean on my
existing files" instruction) - this is the same two-step render-then-tokenize
process OPSD's own data_collator.py uses (render the chat template to a
STRING, then tokenize that string separately, so `max_length` truncation
applies to the whole rendered prompt), duplicated here in a few lines rather
than imported. `tropic/data.py` itself is never imported by this file.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    content: str,
    enable_thinking: bool | None = None,
    max_length: int | None = None,
) -> torch.Tensor:
    messages = [{"role": "user", "content": content}]
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = tokenizer(text, truncation=max_length is not None, max_length=max_length, return_tensors="pt")
    return encoded["input_ids"].squeeze(0)
