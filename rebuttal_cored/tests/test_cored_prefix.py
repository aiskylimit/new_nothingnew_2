import json
import sys
from types import SimpleNamespace

import pytest

from cored.proposal_v2 import prefix_truncate as prefix


@pytest.mark.parametrize('steps', [0, 1, 2, 3, 5, 10, 100])
def test_only_distinct_strictly_shorter_prefixes_are_tested(steps):
    lengths = prefix.candidate_prefix_lengths(steps)
    assert lengths == sorted(set(lengths))
    assert all(0 < length < steps for length in lengths)
    assert len(lengths) <= 5


@pytest.mark.parametrize('passing_length', [None, 3])
@pytest.mark.parametrize('single_only', [False, True])
def test_selection_and_full_fallback_preserve_reasoning_without_testing_full(tmp_path, monkeypatch, passing_length, single_only):
    source, output = tmp_path/'source.jsonl', tmp_path/'prefix.jsonl'
    examples = [{'id': 'long', 'question': 'Q', 'answer': 'A', 'steps': [str(i) for i in range(10)]},
                {'id': 'single', 'question': 'Q', 'answer': 'A', 'steps': ['only']}]
    if single_only:
        examples = examples[1:]
    source.write_text(''.join(json.dumps(row)+'\n' for row in examples))
    prompts_seen = []
    class Engine:
        def __init__(self, **kwargs):
            assert not single_only, 'Full fallback alone must not load an inference model'
        def generate(self, prompts, params):
            prompts_seen.extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text='yes' if int(p) == passing_length else 'no')
                                             for _ in range(4)]) for p in prompts]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'core.local_assets', SimpleNamespace(local_model_path=lambda p: p))
    monkeypatch.setitem(sys.modules, 'core.model_utils', SimpleNamespace(
        load_tokenizer=lambda p: lambda *a, **kw: {'input_ids': [1]}))
    monkeypatch.setitem(sys.modules, 'eval.grade', SimpleNamespace(grade_one=lambda response, gold: response == 'yes'))
    monkeypatch.setattr(prefix, 'build_prefix_prompt', lambda tokenizer, question, steps, k: str(k))
    monkeypatch.delenv('REBUTTAL_STAGE_DIR', raising=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(sys, 'argv', ['prefix', '--in', str(source), '--out', str(output)])
    prefix.main()
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    if single_only:
        assert prompts_seen == [] and actual[0]['steps'] == ['only'] and not actual[0]['prefix_sufficient']
        return
    assert prompts_seen == ['2', '3', '5', '6', '8']
    assert actual[0]['steps'] == examples[0]['steps'][:passing_length or 10]
    assert actual[0]['prefix_sufficient'] == (passing_length is not None)
    assert actual[1]['steps'] == ['only'] and not actual[1]['prefix_sufficient']
