"""Use managed downloads directly, without requiring optional snapshot metadata."""
import json
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rebuttal.common import MODELS, jsonl, rows, sha, write
from rebuttal import data, preflight, provision


def test_download_prepare_and_preflight_accept_assets_without_snapshot_json(tmp_path, monkeypatch):
    assets, prepared, runs = (tmp_path / name for name in ('assets', 'prepared', 'runs'))
    runs.mkdir()

    def parquet(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(values), path)

    for repo, _ in MODELS.values():
        model = assets / 'models' / repo.split('/')[-1]
        write(model / 'config.json', {})
        write(model / 'tokenizer_config.json', {})
    parquet(assets / 'datasets/s1K-1.1/data/train-00000.parquet', [
        dict(question=f'Q {i}', solution=str(i), deepseek_thinking_trajectory='D reasoning',
             deepseek_attempt='D answer', gemini_thinking_trajectory='G reasoning', gemini_attempt='G answer')
        for i in range(1000)])
    benchmarks = assets / 'benchmarks'
    parquet(benchmarks / 'AIME_2024/aime_2024_problems.parquet', [
        {'Problem': f'AIME24 {i}', 'Answer': i} for i in range(30)])
    for folder, count in [('aime_2025', 30), ('aimo-validation-amc', 83)]:
        parquet(benchmarks / folder / 'data/train-00000.parquet', [
            {'problem': f'{folder} {i}', 'answer': str(i)} for i in range(count)])
    jsonl(benchmarks / 'MATH-500/test.jsonl', [
        {'problem': f'MATH {i}', 'answer': str(i)} for i in range(500)])
    parquet(benchmarks / 'ARC-Challenge/test-00000-of-00001.parquet', [
        dict(id=str(i), question=f'ARC {i}', choices={'label': ['A', 'B'], 'text': ['x', 'y']}, answerKey='A')
        for i in range(1172)])
    jsonl(benchmarks / 'LogiQA2.0/test.txt', [
        dict(id=str(i), text='Passage', question=f'LogiQA {i}', options=['a', 'b', 'c', 'd'], answer=0)
        for i in range(1572)])
    monkeypatch.setattr(provision, 'URLS', {
        relative: (url, sha(benchmarks / relative)) for relative, (url, _) in provision.URLS.items()})

    def unexpected_download(*args, **kwargs):
        pytest.fail('Existing local assets must not be downloaded again')

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=unexpected_download))
    monkeypatch.setattr(provision.urllib.request, 'urlretrieve', unexpected_download)
    original_files = {str(p.relative_to(assets)): sha(p) for p in assets.rglob('*') if p.is_file()}
    monkeypatch.setattr(sys, 'argv', ['provision', '--assets', str(assets)])
    provision.main()
    monkeypatch.setattr(sys, 'argv', ['prepare', '--assets', str(assets), '--out', str(prepared), '--skip-statistics'])
    data.main()
    manifest = json.loads((prepared / 'manifest.json').read_text())
    assert manifest['counts'] == {'original': 1000, 'gemini': 1000}
    assert len(rows(prepared / 'original/canonical.jsonl')) == 1000
    assert len(json.loads((prepared / 'benchmarks/manifest.json').read_text())) == 6
    # Statistics generation needs real model tokenizers; isolate only that part here.
    for variant, size in [('original', '7b'), ('gemini', '7b'), ('original', '1.5b')]:
        write(prepared / 'statistics' / f'{variant}_{size}.json', {'data_sha256': manifest['files'][variant]})
    result = preflight.validate(assets, prepared, runs)
    assert result['model_snapshot_verification'] == {'7b': 'not_provided', '1.5b': 'not_provided'}
    assert original_files == {str(p.relative_to(assets)): sha(p) for p in assets.rglob('*') if p.is_file()}
    assert not list(assets.rglob('snapshot.json'))


@pytest.mark.parametrize('contents', ['missing', 'empty', 'cache_only'])
def test_missing_metadata_does_not_accept_missing_assets(tmp_path, contents):
    folder = tmp_path / 'dataset'
    if contents != 'missing':
        folder.mkdir()
    if contents == 'cache_only':
        write(folder / '.cache/huggingface/metadata.json', {})
    with pytest.raises(FileNotFoundError, match='Local model/dataset folder missing|No local model/dataset files'):
        provision.verify_snapshot(folder, 'repo/dataset', 'revision')
