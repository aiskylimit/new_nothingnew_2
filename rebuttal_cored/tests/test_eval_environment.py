import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rebuttal.common import ROOT, write


def test_eval_python_is_independent_of_native_palign(monkeypatch, tmp_path):
    from rebuttal.runner import python_for
    python = tmp_path/'eval-python'
    python.touch()
    python.chmod(0o755)
    monkeypatch.setenv('REBUTTAL_EVAL_PYTHON', str(python))
    monkeypatch.setenv('REBUTTAL_PYTHON_PALIGN', '/missing/native/palign')
    assert python_for('eval') == str(python)
    with pytest.raises(FileNotFoundError, match='REBUTTAL_PYTHON_PALIGN'):
        python_for('palign')


def test_every_method_dispatches_eval_with_the_shared_interpreter(tmp_path, monkeypatch):
    from rebuttal import runner
    calls = []
    def python_for(method):
        calls.append(method)
        assert method == 'eval'
        return '/shared/eval/python'
    monkeypatch.setattr(runner, 'python_for', python_for)
    for method in ('longcot','sgl','ssft','palign','cored'):
        root = tmp_path/method
        write(root/'final_checkpoint.json', {'path':'/saved/checkpoint'})
        job = object.__new__(runner.Job)
        job.root, job.data, job.gpus, job.config = root, tmp_path/'data', '0,1', {}
        commands = []
        job.execute = lambda name, args, count, **kwargs: commands.append((name,args,count))
        job.evaluate()
        assert [name for name, _, _ in commands] == ['evaluation_generate','evaluation_score']
        assert commands[0][2] == 2
        assert all(command[0] == '/shared/eval/python' for _, pairs, _ in commands for command, _ in pairs)
    assert calls == ['eval']*5


def test_zero_adapter_has_real_qproj_shapes_and_no_backbone_weights(tmp_path):
    from rebuttal.eval_probe import zero_adapter
    from safetensors.torch import load_file
    config = SimpleNamespace(hidden_size=32, num_attention_heads=4, head_dim=8, num_hidden_layers=2)
    value = zero_adapter(config, Path('/local/model'), tmp_path)
    assert value['r'] == 16 and value['target_modules'] == ['q_proj']
    weights = load_file(str(tmp_path/'adapter_model.safetensors'))
    assert len(weights) == 4
    for key, tensor in weights.items():
        assert tensor.shape == ((16,32) if '.lora_A.' in key else (32,16))
        assert tensor.count_nonzero().item() == 0


@pytest.mark.parametrize('failure', [False, True])
def test_probe_writes_pass_or_failure_from_actual_engine_call(tmp_path, monkeypatch, failure):
    from rebuttal import eval_probe
    monkeypatch.setattr(sys, 'argv', ['probe','--model',str(tmp_path/'model'),'--out',str(tmp_path/'probe.json')])
    monkeypatch.setattr(eval_probe, 'cuda_runtime', lambda: {'torch_cuda':'12.9','gpu':'fixture'})
    monkeypatch.setattr(eval_probe.importlib.metadata, 'version', lambda _: 'fixture')
    monkeypatch.setattr(eval_probe, 'zero_adapter', lambda *a: {'r':16})
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: object()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object())))
    requests = []
    class Engine:
        def generate(self, prompts, params, **kwargs):
            requests.append(kwargs['lora_request'])
            if failure:
                raise ImportError('libcudart.so.13 fixture failure')
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1])])]
    def engine(*a, **kwargs):
        assert a[1] == 2048 and kwargs['enforce_eager'] and kwargs['max_num_seqs'] == 1
        return Engine()
    monkeypatch.setattr(eval_probe, 'load_backend', lambda: SimpleNamespace(
        LLM=Engine, _make_llm=engine, apply_chat=lambda *a: 'prompt'))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'vllm.lora.request', SimpleNamespace(LoRARequest=lambda *a: a))
    if failure:
        with pytest.raises(ImportError, match='libcudart'):
            eval_probe.main()
    else:
        eval_probe.main()
    value = json.loads((tmp_path/'probe.json').read_text())
    assert value['status'] == ('failed' if failure else 'passed')
    assert requests and requests[0][0] == 'preflight-zero'


def test_eval_preflight_uses_eval_python_each_gpu_and_no_training_telemetry(tmp_path, monkeypatch):
    from rebuttal import eval_preflight, runner
    monkeypatch.setattr(runner, 'python_for', lambda method: '/eval/python' if method == 'eval' else pytest.fail('Wrong env'))
    monkeypatch.setenv('REBUTTAL_STAGE_DIR', '/training/telemetry')
    monkeypatch.setenv('RANK', '7')
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command,kwargs))
        def poll(self): return 0
    monkeypatch.setattr(eval_preflight.subprocess, 'Popen', Process)
    eval_preflight.check_eval(tmp_path/'assets', tmp_path/'runs', '0,1', sizes=('1.5b',))
    assert len(calls) == 2
    assert [kwargs['env']['CUDA_VISIBLE_DEVICES'] for _, kwargs in calls] == ['0','1']
    for command, kwargs in calls:
        assert command[0] == '/eval/python' and 'rebuttal.eval_probe' in command
        assert 'RANK' not in kwargs['env'] and 'REBUTTAL_STAGE_DIR' not in kwargs['env']


def test_eval_manifest_uses_package_pins_without_direct_urls_or_training_frameworks():
    from packaging.requirements import Requirement
    text = (ROOT/'rebuttal_eval.txt').read_text()
    requirements = [Requirement(line) for line in text.splitlines() if line and not line.startswith('#')]
    assert all(requirement.url is None for requirement in requirements)
    for package, version in [('torch','2.11.0'),('torchvision','0.26.0'),
                             ('torchaudio','2.11.0'),('vllm','0.26.0')]:
        assert f'{package}=={version}' in text.splitlines()
    assert 'llamafactory' not in text and 'unsloth' not in text


@pytest.mark.parametrize('name', ['rebuttal', 'sft', 'palign', 'sgl',
                                  'SegmentSelectiveSFT', 'cored', 'rebuttal_eval'])
def test_environment_manifests_use_plain_pypi_pins_with_matching_gpu_packages(name):
    from packaging.requirements import Requirement
    from packaging.version import Version
    text = (ROOT/f'{name}.txt').read_text()
    requirements = [Requirement(line) for line in text.splitlines()
                    if line and not line.startswith('#')]
    assert all(requirement.url is None for requirement in requirements)
    for requirement in requirements:
        assert all(Version(pin.version).local is None for pin in requirement.specifier)
    # These exact families match the default Linux PyPI wheels and vLLM metadata;
    # a source override or a lone Torch upgrade would reintroduce the server failure.
    pins = {r.name: str(r.specifier) for r in requirements}
    if name in {'sgl', 'SegmentSelectiveSFT'}:
        assert pins['torch'] == '==2.9.0' and pins['torchvision'] == '==0.24.0'
        assert 'vllm' not in pins
    else:
        assert pins['torch'] == '==2.11.0' and pins['torchvision'] == '==0.26.0'
        assert pins['torchaudio'] == '==2.11.0' and pins['vllm'] == '==0.26.0'
        assert Version(pins['transformers'].removeprefix('==')) >= Version('5.5.3')
    if name == 'SegmentSelectiveSFT':
        assert pins['xformers'] == '==0.0.33.post1' and pins['torchao'] == '==0.14.1'
