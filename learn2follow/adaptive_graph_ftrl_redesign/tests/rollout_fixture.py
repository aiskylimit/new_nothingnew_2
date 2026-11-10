"""Instrumented vLLM substitute; never downloads a model."""

from types import SimpleNamespace

import torch


class FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.removed = []
        self.llm_engine = self
        self.asleep = False
        self.fail = False
        torch.rand(5)

    def remove_lora(self, adapter):
        self.removed.append(adapter)

    def generate(self, prompts, sampling_params, lora_request, use_tqdm):
        assert not self.asleep
        self.calls.append((prompts, sampling_params, lora_request))
        torch.rand(5)
        if self.fail:
            raise RuntimeError("generation failed")
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[14, 2])]) for _ in prompts]

    def sleep(self, level):
        assert level == 1
        self.asleep = True

    def wake_up(self):
        assert self.asleep
        self.asleep = False

def request(name, number, path):
    return SimpleNamespace(lora_int_id=number, lora_path=path, name=name)
