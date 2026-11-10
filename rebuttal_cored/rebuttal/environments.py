"""Select provisioned interpreters and keep their library search paths separate."""
import os
from pathlib import Path

from rebuttal.common import ROOT

FOLDERS = {'rebuttal': 'rebuttal', 'longcot': 'sft', 'palign': 'palign', 'sgl': 'sgl',
           'ssft': 'SegmentSelectiveSFT', 'cored': 'cored', 'eval': 'rebuttal_eval'}
PROJECT_PATHS = (ROOT / 'sgl/src', ROOT, ROOT / 'cored')


def python_for(method):
    folder = FOLDERS[method]
    root = Path(os.environ.get('REBUTTAL_ENV_ROOT', '/mnt/local/uvenvs'))
    variable = ('REBUTTAL_EVAL_PYTHON' if method == 'eval' else 'REBUTTAL_PYTHON'
                if method == 'rebuttal' else 'REBUTTAL_PYTHON_' + method.upper())
    env_variable = {'rebuttal': 'REBUTTAL_VENV', 'eval': 'REBUTTAL_EVAL_VENV'}.get(method)
    default = Path(os.environ.get(env_variable, str(root / folder))) if env_variable else root / folder
    value = os.path.abspath(os.path.expanduser(os.environ.get(variable, str(default / 'bin/python'))))
    # Do not resolve symlinks: a venv's Python may point at the shared base Python.
    if not Path(value).is_file() or not os.access(value, os.X_OK):
        raise FileNotFoundError(f'Missing Python for {method}: {value}. '
                                f'Server environment: {folder}.txt; override: {variable}')
    return value


def environment_for(python, base=None):
    env = dict(os.environ if base is None else base)
    binary = Path(os.path.abspath(python)).parent
    target = binary.parent
    env_root = Path(os.path.abspath(env.get('REBUTTAL_ENV_ROOT', '/mnt/local/uvenvs')))
    inherited = [Path(os.path.abspath(env[name])) for name in ('VIRTUAL_ENV', 'CONDA_PREFIX') if env.get(name)]

    def foreign(path):
        path = Path(path)
        if path.is_relative_to(target):
            return False
        return path.is_relative_to(env_root) or any(path.is_relative_to(root) for root in inherited)

    paths = [p for p in env.get('PATH', '').split(os.pathsep) if p and p != str(binary) and not foreign(p)]
    env['PATH'] = os.pathsep.join([str(binary), *paths])
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = os.pathsep.join(
            p for p in env['LD_LIBRARY_PATH'].split(os.pathsep) if p and not foreign(p))
    for name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONUSERBASE', 'PYTHONSTARTUP', 'PYTHONEXECUTABLE',
                 '__PYVENV_LAUNCHER__', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER',
                 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env.update(VIRTUAL_ENV=str(target), PYTHONNOUSERSITE='1',
               PYTHONPATH=os.pathsep.join(map(str, PROJECT_PATHS)))
    return env
