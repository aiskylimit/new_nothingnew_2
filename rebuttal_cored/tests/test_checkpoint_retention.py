import pytest

from rebuttal.checkpoints import keep_final_model
from rebuttal.common import sha, write


def put(path, data=b'intermediate'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def final_model(run, method):
    relative = ('artifacts/cored/distill/student' if method == 'cored' else
                'checkpoint/checkpoint-final' if method == 'ssft' else 'checkpoint')
    final = run / relative
    weight = put(final/'adapter_model.safetensors', b'final weights')
    write(final/'adapter_config.json', {'base_model_name_or_path': '/immutable/base'})
    write(final/'checkpoint_manifest.json', {'run_id': 'fixture', 'files_sha256': {
        p.name: sha(p) for p in final.iterdir() if p.is_file()}})
    return final, weight


@pytest.mark.parametrize('method', ['longcot', 'palign', 'sgl', 'ssft'])
def test_finished_run_retains_only_final_model_and_preserves_logs_and_other_runs(tmp_path, method):
    run = tmp_path/'run'
    final, weight = final_model(run, method)
    previous = [put(run/f'checkpoint/checkpoint-{step}/optimizer.pt') for step in (32, 64, 96)]
    retained = [put(run/'logs/train.log'), put(run/'training_cost.json'),
                put(run/'artifacts/cored/phase1/timing-rank-0.json'),
                put(run/'artifacts/cored/weights/sample.npz'),
                put(tmp_path/'other-run/checkpoint/checkpoint-32/optimizer.pt')]
    original = {p: p.read_bytes() for p in final.iterdir() if p.is_file()}
    assert keep_final_model(run, method, final)
    assert all(not p.exists() for p in previous)
    assert all(p.exists() for p in retained)
    assert {p: p.read_bytes() for p in final.iterdir() if p.is_file()} == original
    assert weight.read_bytes() == b'final weights'
    assert keep_final_model(run, method, final) == []


def test_cored_retains_all_models_resume_checkpoints_and_artifacts(tmp_path):
    final, _ = final_model(tmp_path, 'cored')
    for i in range(3):
        put(tmp_path/f'artifacts/cored/phase1/expert_{i}/adapter_model.safetensors')
    for name in ('merge_ties/merged/adapter_model.safetensors', 'distill/teacher/adapter_model.safetensors',
                 'phase1/resume.pt', 'distill/resume.pt', 'weights/sample.npz', 'phase1/timing-rank-0.json'):
        put(tmp_path/'artifacts/cored'/name)
    put(tmp_path/'checkpoint/checkpoint-32/optimizer.pt')
    original = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert keep_final_model(tmp_path, 'cored', final) == []
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == original


@pytest.mark.parametrize('problem', ['missing', 'corrupt', 'unlisted'])
def test_invalid_final_model_never_deletes_resume_checkpoints(tmp_path, problem):
    final, weight = final_model(tmp_path, 'ssft')
    resume = put(tmp_path/'checkpoint/checkpoint-32/optimizer.pt')
    if problem == 'missing':
        weight.unlink()
    elif problem == 'corrupt':
        weight.write_bytes(b'corrupt')
    else:
        write(final/'checkpoint_manifest.json', {'run_id': 'fixture', 'files_sha256': {}})
    with pytest.raises(ValueError):
        keep_final_model(tmp_path, 'ssft', final)
    assert resume.exists()


def test_cleanup_rejects_checkpoint_symlinks_without_deleting_anything(tmp_path):
    run = tmp_path/'run'
    final, _ = final_model(run, 'longcot')
    resume = put(run/'checkpoint/checkpoint-32/optimizer.pt')
    external = put(tmp_path/'another-project/adapter_model.safetensors')
    (run/'checkpoint/checkpoint-64').symlink_to(external.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='Unsafe'):
        keep_final_model(run, 'longcot', final)
    assert external.exists() and resume.exists()


def test_cleanup_requires_expected_final_path(tmp_path):
    final, _ = final_model(tmp_path, 'ssft')
    resume = put(tmp_path/'checkpoint/checkpoint-32/optimizer.pt')
    with pytest.raises(ValueError, match='Unexpected'):
        keep_final_model(tmp_path, 'longcot', final)
    assert resume.exists()
