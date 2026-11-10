"""Load the unchanged Qwen/LoRA training recipe."""

import torch


def load_runtime(cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, cfg.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        raise RuntimeError("CPU real-model smoke requires dtype=float32")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=dtype, attn_implementation=cfg.attention_backend, use_cache=False
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model, tokenizer
