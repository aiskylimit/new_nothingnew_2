"""answer_gain with vLLM's prefix cache on: the returned prompt_logprobs cover only the computed tail."""

import json
import sys
import types

import pytest

from sgl.signals import answer_gain


class Tokenizer:
    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


@pytest.fixture
def fake_vllm(monkeypatch):
    """A vLLM that caches whole 4-token blocks and returns rows only for what it computed."""
    seen = []

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, params):
            outputs = []
            for prompt in prompts:
                ids = prompt["prompt_token_ids"]
                hit = 0
                if not params.skip_reading_prefix_cache:
                    for other in seen:
                        common = 0
                        while common < min(len(ids) - 1, len(other)) and ids[common] == other[common]:
                            common += 1
                        hit = max(hit, common // 4 * 4)
                seen.append(ids)
                # logprob of a token = -(its position), so a shifted read changes the result
                rows = [{ids[i]: types.SimpleNamespace(logprob=-float(i))} for i in range(hit, len(ids))]
                outputs.append(types.SimpleNamespace(prompt_logprobs=rows))
            return outputs

    vllm = types.SimpleNamespace(LLM=LLM, SamplingParams=Params)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", types.SimpleNamespace(TokensPrompt=lambda **kw: kw))
    auto = types.SimpleNamespace(from_pretrained=lambda name: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=auto))
    return seen


def run(tmp_path, name, *extra):
    record = {
        "id": 0, "response": "\\boxed{1234567890}", "input_ids": list(range(40)),
        "response_token_span": [8, 40],
        # the repeated cut makes two identical probes: the second is cached past the answer start
        "steps": [{"token_end": 20}, {"token_end": 20}, {"token_end": 40}],
    }
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(record) + "\n")
    output, meta = tmp_path / f"{name}.json", tmp_path / f"{name}.meta.json"
    answer_gain.main(["--data-path", str(data), "--model", "m", "--output", str(output),
                      "--meta-output", str(meta), *extra])
    return json.loads(output.read_text()), json.loads(meta.read_text())


def test_cached_scoring_matches_uncached(tmp_path, fake_vllm):
    cached, meta = run(tmp_path, "cached")
    fake_vllm.clear()
    uncached, _ = run(tmp_path, "uncached", "--no-prefix-cache")
    assert cached == uncached
    assert meta["n_rescored"] == 1


def test_misaligned_rows_raise():
    rows = [{7: types.SimpleNamespace(logprob=-1.0)}]
    with pytest.raises(KeyError):
        answer_gain.answer_logp(rows, [8], 1)
