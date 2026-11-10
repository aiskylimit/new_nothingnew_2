import json
import os
import sys
import venv

import pytest

from rebuttal.common import ROOT, matrix, write
from rebuttal.environments import FOLDERS, PROJECT_PATHS, environment_for, python_for


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    root = tmp_path/'envs'
    for key in list(os.environ):
        if key.startswith('REBUTTAL_PYTHON') or key in ('REBUTTAL_EVAL_PYTHON', 'REBUTTAL_EVAL_VENV', 'REBUTTAL_VENV'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('REBUTTAL_ENV_ROOT', str(root))
    for folder in FOLDERS.values():
        executable = root/folder/'bin/python'
        executable.parent.mkdir(parents=True)
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
    return root


def test_all_seven_txt_files_resolve_to_their_own_interpreters(provisioned, monkeypatch):
    monkeypatch.setenv('VIRTUAL_ENV', '/foreign/env')
    monkeypatch.setenv('PY', '/foreign/env/bin/python')
    for method, folder in FOLDERS.items():
        assert (ROOT/f'{folder}.txt').is_file()
        assert python_for(method) == str(provisioned/folder/'bin/python')
    (provisioned/'sft/bin/python').unlink()
    with pytest.raises(FileNotFoundError, match='sft.txt'):
        python_for('longcot')


def test_worker_environment_removes_foreign_packages_but_keeps_cuda_and_project_code(provisioned):
    target = provisioned/'SegmentSelectiveSFT'
    foreign = provisioned/'rebuttal'
    base = dict(os.environ, VIRTUAL_ENV=str(foreign), CONDA_PREFIX='/other/conda',
                PATH=f'{foreign}/bin:{provisioned}/cored/bin:/other/conda/bin:/usr/bin',
                PYTHONPATH=f'{foreign}/lib/site-packages', PYTHONHOME=str(foreign),
                PYTHONUSERBASE='/other/user-packages', RANK='7', WORLD_SIZE='8',
                LD_LIBRARY_PATH=f'{foreign}/lib:{target}/lib:/usr/local/cuda/lib64',
                CUDA_VISIBLE_DEVICES='0,1,2,3')
    result = environment_for(target/'bin/python', base)
    assert result['VIRTUAL_ENV'] == str(target)
    assert result['PATH'] == f'{target}/bin:/usr/bin'
    assert result['PYTHONPATH'] == os.pathsep.join(map(str, PROJECT_PATHS))
    assert result['LD_LIBRARY_PATH'] == f'{target}/lib:/usr/local/cuda/lib64'
    assert result['PYTHONNOUSERSITE'] == '1' and result['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert all(key not in result for key in ('PYTHONHOME', 'PYTHONUSERBASE', 'CONDA_PREFIX', 'RANK', 'WORLD_SIZE'))
    assert base['VIRTUAL_ENV'] == str(foreign)


def test_actual_worker_imports_only_the_selected_venv_package(tmp_path, monkeypatch):
    from rebuttal.runner import Job
    root = tmp_path/'envs'
    target, foreign = root/'sft', root/'rebuttal'
    for folder, marker in ((target, 'sft'), (foreign, 'rebuttal')):
        venv.EnvBuilder(with_pip=False).create(folder)
        site = folder/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
        site.mkdir(parents=True, exist_ok=True)
        (site/'env_routing_fixture.py').write_text(f'MARKER = {marker!r}\n')
    monkeypatch.setenv('REBUTTAL_ENV_ROOT', str(root))
    monkeypatch.setenv('VIRTUAL_ENV', str(foreign))
    monkeypatch.setenv('PYTHONPATH', str(foreign/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'))
    monkeypatch.setenv('PYTHONHOME', str(foreign))
    monkeypatch.setenv('PATH', str(foreign/'bin')+os.pathsep+os.environ['PATH'])
    job = object.__new__(Job)
    job.root, job.config, job.gpus, job.fingerprint = tmp_path, {'run_id': 'env-test'}, '', 'env-test'
    command = [str(target/'bin/python'), '-c',
               'import env_routing_fixture as m,sys,json,os; '
               'print(json.dumps(dict(marker=m.MARKER,prefix=sys.prefix,venv=os.environ["VIRTUAL_ENV"])))']
    job.execute('import_test', [(command, '')], 0, instrument=False)
    record = json.loads(next((tmp_path/'logs').glob('*.log')).read_text())
    assert record == {'marker': 'sft', 'prefix': str(target), 'venv': str(target)}


@pytest.mark.parametrize('method', ['longcot', 'sgl', 'ssft', 'palign', 'cored'])
def test_native_preprocessing_and_training_start_in_the_method_environment(tmp_path, monkeypatch, provisioned, method):
    from rebuttal import runner
    from rebuttal.smoke_2gpu import fixtures
    fixtures(tmp_path/'data')
    config = next(c for c in matrix() if c['method'] == method)
    write(tmp_path/'assets/models'/config['model']['repo'].split('/')[-1]/'config.json', {})
    target = provisioned/FOLDERS[method]
    def metadata(command, **kwargs):
        if command[0] == 'git':
            return 'fixture'
        assert command[0] == str(target/'bin/python')
        assert kwargs['env']['VIRTUAL_ENV'] == str(target)
        return '{}'
    monkeypatch.setattr(runner.subprocess, 'check_output', metadata)
    job = runner.Job(config, tmp_path/'assets', tmp_path/'data', tmp_path/'runs', '0,1,2,3')
    class Dispatched(Exception):
        pass
    def execute(stage, commands, count):
        assert count == 4
        assert all(command[0] == str(target/'bin/python') for command, _ in commands)
        raise Dispatched
    monkeypatch.setattr(job, 'execute', execute)
    with pytest.raises(Dispatched):
        job.train()
