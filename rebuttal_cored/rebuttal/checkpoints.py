"""Remove intermediate model checkpoints only after the final model is saved."""
import json
import re
import shutil
from pathlib import Path

from rebuttal.common import sha


def keep_final_model(run, method, final):
    # CoReD retains every artifact produced by its native pipeline.
    if method == 'cored':
        return []
    run, final = Path(run).resolve(), Path(final).resolve()
    expected = run / ('checkpoint/checkpoint-final' if method == 'ssft' else 'checkpoint')
    if final != expected or not list(final.glob('*.safetensors')):
        raise ValueError(f'Unexpected or missing final model: {final}')
    manifest = json.loads((final / 'checkpoint_manifest.json').read_text())
    if not {p.name for p in final.glob('*.safetensors')} <= manifest['files_sha256'].keys():
        raise ValueError(f'Final model manifest is incomplete: {final}')
    for name, checksum in manifest['files_sha256'].items():
        if Path(name).name != name or sha(final / name) != checksum:
            raise ValueError(f'Final model verification failed: {final}/{name}')

    candidates = [p for p in (run / 'checkpoint').glob('checkpoint-*')
                  if re.fullmatch(r'checkpoint-\d+', p.name)]
    candidates = [p for p in candidates if p.exists() or p.is_symlink()]
    # Validate the deletion plan first; never follow a checkpoint symlink out of
    # the run or remove a parent of the final model.
    for path in candidates:
        if (path.is_symlink() or not path.resolve().is_relative_to(run)
                or final.is_relative_to(path.resolve())):
            raise ValueError(f'Unsafe intermediate checkpoint path: {path}')
    for path in candidates:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if candidates:
        print(f'[{manifest["run_id"]}] kept final model; removed {len(candidates)} intermediate checkpoints', flush=True)
    return candidates
