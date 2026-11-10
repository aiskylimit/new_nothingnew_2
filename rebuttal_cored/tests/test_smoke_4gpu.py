import copy
import json
import sys
from types import SimpleNamespace

import pytest

from rebuttal.common import MATH, OOD, jsonl, matrix, rows, write
from rebuttal.data import adapt
from rebuttal.smoke import configuration, prepare_subset, verify_run


def local_data(root):
    for variant in ('original', 'gemini'):
        values = [dict(id=str(i), question=f'Q{i}', thinking='First step.\n\nSecond step.',
                       answer_text='Answer 2', gold='2', teacher=variant) for i in range(10)]
        values[0]['thinking'] = 'one segment'
        values[1]['thinking'] = ('long ' * 5000) + '\n\nSecond step.'
        adapt(values, root/variant)
    for name in (*MATH, *OOD):
        jsonl(root/'benchmarks'/f'{name}.jsonl', [dict(question_id=f'{name}:{i}', question='Q', answer='2')
                                               for i in range(8)])
    write(root/'manifest.json', {'source': 'local fixture representing prepared real-data schema'})


def test_smoke_profiles_preserve_main_settings_and_all_method_filters():
    originals = matrix()
    unchanged = copy.deepcopy(originals)
    configs = [configuration(c) for c in originals if c['setting'] == 'C']
    assert originals == unchanged and len(configs) == 5
    for config in configs:
        assert config['smoke_steps'] == 4 and config['smoke_samples'] == 8
        assert config['train']['global_batch'] == 4 and config['train']['micro_batch'] == 1
        assert config['development_smoke']['world_size'] == 4
        assert config['train']['cutoff'] == 2048 and config['development_smoke']['context'] == 4096
        if config['method'] == 'cored':
            assert config['method_specific']['M'] == 3 and config['train']['epochs'] == 3
        if config['method'] == 'palign':
            assert config['method_specific']['filter'] == 'verified_continuation'
        if config['method'] == 'ssft':
            assert config['method_specific']['require_valid_attribution']


def test_smoke_selects_real_short_examples_and_all_benchmarks_without_changing_source(tmp_path, monkeypatch):
    source, destination = tmp_path/'prepared', tmp_path/'smoke-inputs'
    local_data(source)
    original = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    tokenizer = SimpleNamespace(encode=lambda text, **kw: text.split())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **kw: tokenizer)))
    prepare_subset(tmp_path/'model', source, destination)
    for variant in ('original', 'gemini'):
        assert [r['id'] for r in rows(destination/variant/'canonical.jsonl')] == list(map(str, range(2, 10)))
    for name in (*MATH, *OOD):
        assert len(rows(destination/'benchmarks'/f'{name}.jsonl')) == 4
    assert original == {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert prepare_subset(tmp_path/'model', source, destination) == destination
    (destination/'original/canonical.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Smoke input changed'):
        prepare_subset(tmp_path/'model', source, destination)


@pytest.mark.parametrize('method', ['longcot', 'sgl', 'ssft', 'palign', 'cored'])
def test_smoke_dispatches_every_native_stage_on_four_gpus(tmp_path, monkeypatch, method):
    from rebuttal import runner, method_prep
    local_data(tmp_path/'data')
    config = configuration(next(c for c in matrix() if c['method'] == method))
    model = tmp_path/'assets/models'/config['model']['repo'].split('/')[-1]
    write(model/'config.json', {})
    monkeypatch.setattr(runner, 'python_for', lambda _: sys.executable)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: '{}')
    monkeypatch.setattr(method_prep, 'merge_ssft', lambda *a: None)
    job = runner.Job(config, tmp_path/'assets', tmp_path/'data', tmp_path/'runs', '0,1,2,3')
    seen = []
    class Captured(Exception):
        pass
    def execute(stage, commands, count):
        seen.append(stage)
        assert count == 4
        if stage == 'prefix_alignment':
            assert len(commands) == 4
            values = rows(job.input/'canonical.jsonl')
            for rank in range(4):
                jsonl(job.root/f'palign-{rank}.jsonl', [dict(r, accepted=True, prefix=r['thinking'], continuation='2')
                                                     for r in values[rank::4]])
            return
        if stage == 'attribution':
            assert len(commands) == 4 and config['method_specific']['ig_steps'] == 4
            return
        command = commands[0][0]
        if stage == 'spectral_masks':
            assert '--nproc_per_node=4' in command
            assert command[command.index('--max-length')+1] == '2048'
            return
        if method == 'cored':
            assert '--smoke_4gpu' in command and '--development_smoke' not in command
            assert command[command.index('--dry_run')+1] == 4
            assert command[command.index('--global_batch')+1] == 4
            assert command[command.index('--smoke_context')+1] == 4096
            if stage == 'cored_smoke_phase1':
                assert command[-4:] == ['prepare', 'prefix', 'key', 'phase1']
                return
            assert stage == 'cored_smoke_student' and command[-3:] == ['weights', 'merge', 'distill']
        else:
            assert stage == 'optimization' and '--nproc_per_node=4' in command
            if method == 'sgl':
                assert command[command.index('--max-steps')+1] == '4'
            elif method == 'ssft':
                assert command[command.index('--max_steps')+1] == '4'
            else:
                import yaml
                settings = yaml.safe_load((job.root/'native_train.yaml').read_text())
                assert settings['max_steps'] == 4 and settings['per_device_train_batch_size'] == 1
        raise Captured
    monkeypatch.setattr(job, 'execute', execute)
    with pytest.raises(Captured):
        job.train()
    assert seen[-1] == ('cored_smoke_student' if method == 'cored' else 'optimization')


def test_smoke_runner_defaults_to_five_small_model_runs(tmp_path, monkeypatch):
    from rebuttal import runner, preflight, smoke
    monkeypatch.setattr(sys, 'argv', ['runner', 'smoke', '--runs', str(tmp_path/'smoke')])
    validations, events = [], []
    monkeypatch.setattr(preflight, 'validate', lambda *a, **kw: validations.append(kw))
    monkeypatch.setattr(smoke, 'prepare_subset', lambda *a: tmp_path/'small-data')
    monkeypatch.setattr(runner, 'aggregate', lambda *a: None)
    monkeypatch.setattr(smoke, 'verify_run', lambda job: events.append(('verify', job.config['run_id'])))
    class FakeJob:
        def __init__(self, config, assets, data, runs, gpus):
            self.config, self.root = config, runs/config['run_id']
            self.root.mkdir()
            assert config['smoke'] and gpus == '0,1,2,3' and data == tmp_path/'small-data'
            assert config['setting'] == 'C' and config['model']['size'] == '1.5b'
        def train(self):
            events.append(('train', self.config['run_id']))
        def evaluate(self):
            events.append(('eval', self.config['run_id']))
    monkeypatch.setattr(runner, 'Job', FakeJob)
    runner.main()
    assert len(validations) == 1 and validations[0]['check_evaluator'] is False
    assert [stage for stage, _ in events] == ['train', 'eval', 'verify'] * 5
    result = json.loads((tmp_path/'smoke/summary.json').read_text())
    assert result['status'] == 'passed' and len(set(result['completed'])) == 5


@pytest.mark.parametrize('method', ['longcot', 'cored'])
@pytest.mark.parametrize('problem', [None, 'nan', 'no_updates', 'incomplete_eval', 'missing_rank'])
def test_smoke_cannot_pass_without_finite_trained_checkpoint_and_scored_eval(tmp_path, problem, method):
    import torch
    from safetensors.torch import save_file
    config = configuration(next(c for c in matrix() if c['method'] == method))
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    save_file({'lora': torch.tensor([float('nan') if problem == 'nan' else 1.0])}, str(checkpoint/'adapter.safetensors'))
    write(tmp_path/'final_checkpoint.json', {'path': str(checkpoint)})
    write(tmp_path/'training_cost.json', {'stages': [{'stage': 'optimization', 'optimization_steps': 0 if problem == 'no_updates' else 4}]})
    if method == 'cored':
        for stage in ('phase1', 'distill'):
            for rank in range(3 if problem == 'missing_rank' else 4):
                write(tmp_path/'artifacts/cored'/stage/f'timing-rank-{rank}.json',
                      {'rank': rank, 'optimizer_steps': 0 if problem == 'no_updates' else 4})
    for name in (*MATH, *OOD):
        questions = [{'question_id': f'{name}:{i}'} for i in range(4)]
        jsonl(tmp_path/'data/benchmarks'/f'{name}.jsonl', questions)
        scored = [dict(row, eval_seed=42 if name in MATH else 0, sample_id=0, is_correct=False, finish_reason='stop')
                  for row in questions]
        jsonl(tmp_path/'evaluation'/f'{name}.scored.jsonl', scored[:-1] if problem == 'incomplete_eval' else scored)
    job = SimpleNamespace(root=tmp_path, data=tmp_path/'data', config=config)
    if problem and (problem != 'missing_rank' or method == 'cored'):
        with pytest.raises(ValueError):
            verify_run(job)
    else:
        verify_run(job)


@pytest.mark.parametrize('problem', ['preflight', 'incomplete'])
def test_failed_smoke_replaces_an_old_pass_with_failure(tmp_path, problem):
    from rebuttal.smoke import smoke_status
    write(tmp_path/'summary.json', {'status': 'passed'})
    with pytest.raises((RuntimeError, ValueError)):
        with smoke_status(tmp_path, ['expected_run']):
            if problem == 'preflight':
                raise RuntimeError('preflight failed')
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['status'] == 'failed' and summary['completed'] == [] and summary['error']


@pytest.mark.parametrize('gpus,flags', [
    ('0,1,2,3', []), ('0,1', ['--dry_run', '4']),
    ('0,1,2,3', ['--dry_run', '4', '--formal_2h100']),
    ('0,1,2,3', ['--dry_run', '4', '--smoke_context', '128']),
])
def test_native_four_gpu_smoke_rejects_wrong_topology_or_unsafe_limits(monkeypatch, gpus, flags):
    from rebuttal.common import ROOT
    monkeypatch.syspath_prepend(str(ROOT/'cored'))
    from train import pipeline_b200
    monkeypatch.setattr(pipeline_b200, 'local_model_path', lambda path: path)
    monkeypatch.setattr(sys, 'argv', ['pipeline', '--gpus', gpus, '--smoke_4gpu', *flags])
    with pytest.raises(ValueError):
        pipeline_b200.main()
