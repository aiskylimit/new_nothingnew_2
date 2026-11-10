"""Four-GPU smoke with real backbones and small, isolated real-data subsets."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

from rebuttal.common import MATH, OOD, jsonl, rows, sha, write
from rebuttal.data import adapt

SAMPLES = 8
STEPS = 4
CUTOFF = 2048
CONTEXT = 4096


@contextmanager
def smoke_status(directory, expected_runs):
    summary = {'status': 'running', 'completed': [], 'expected_runs': expected_runs,
               'scope': 'small real-data smoke; not full-batch/full-context memory validation'}
    path = Path(directory) / 'summary.json'
    write(path, summary)
    try:
        yield summary
        if summary['completed'] != expected_runs:
            raise ValueError('Smoke did not complete every selected run')
    except BaseException as error:
        summary.update(status='failed', error=repr(error))
        write(path, summary)
        raise
    else:
        summary['status'] = 'passed'
        write(path, summary)


def configuration(original):
    config = copy.deepcopy(original)
    config.update(run_id='smoke4_' + original['run_id'], smoke=True,
                  smoke_samples=SAMPLES, smoke_steps=STEPS,
                  development_smoke={'world_size': 4, 'context': CONTEXT, 'new_tokens': 256,
                                     'source': 'short real s1K/benchmark subsets'})
    config['data']['expected_samples'] = SAMPLES
    config['train'].update(global_batch=4, micro_batch=1, gradient_accumulation=1,
                           cutoff=CUTOFF, epochs=3)
    if config['method'] == 'cored':
        config['train']['gradient_accumulation'] = 'adaptive token budget'
    if config['method'] == 'ssft':
        config['method_specific'].update(ig_steps=4, ig_max_tokens=CUTOFF, require_valid_attribution=True)
    config['evaluation'].update(seeds=[42], budgets=[1], context=CONTEXT,
                                math_max_tokens=256, arc_max_tokens=256, logiqa_max_tokens=256)
    return config


def prepare_subset(model, source, destination):
    """Select short real examples by length only, never by answer correctness."""
    from transformers import AutoTokenizer
    model, source, destination = map(Path, (model, source, destination))
    identity = {'model': str(model), 'source_manifest_sha256': sha(source / 'manifest.json'),
                'samples': SAMPLES, 'cutoff': CUTOFF, 'benchmark_samples': 4}
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['identity'] != identity or any(
                sha(destination / name) != digest for name, digest in saved['files_sha256'].items()):
            raise ValueError('Smoke input changed; choose another REBUTTAL_SMOKE_RUNS directory')
        return destination
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True)
    selected_ids = {}
    for variant in ('original', 'gemini'):
        candidates = rows(source / variant / 'canonical.jsonl')
        candidates.sort(key=lambda r: len(r['question']) + len(r['thinking']) + len(r['answer_text']))
        selected = []
        for row in candidates:
            # At least two paragraphs exercise shorter-prefix selection and segment masks.
            if len([part for part in row['thinking'].split('\n\n') if part.strip()]) < 2:
                continue
            text = '\n'.join(str(row[key]) for key in ('question', 'thinking', 'answer_text', 'gold'))
            if len(tokenizer.encode(text, add_special_tokens=False)) > CUTOFF - 512:
                continue
            selected.append(row)
            if len(selected) == SAMPLES:
                break
        if len(selected) != SAMPLES:
            raise ValueError(f'Smoke needs {SAMPLES} short real {variant} examples with at least two paragraphs')
        adapt(selected, destination / variant)
        selected_ids[variant] = [row['id'] for row in selected]
    benchmarks = {}
    for name in (*MATH, *OOD):
        path = source / 'benchmarks' / f'{name}.jsonl'
        examples = rows(path)[:4]
        if len(examples) != 4:
            raise ValueError(f'Smoke needs four real benchmark examples: {path}')
        output = destination / 'benchmarks' / f'{name}.jsonl'
        jsonl(output, examples)
        benchmarks[name] = {'count': 4, 'sha256': sha(output), 'source_sha256': sha(path), 'smoke': True}
    write(destination / 'benchmarks/manifest.json', benchmarks)
    files = {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*')
             if p.is_file() and p != manifest_path}
    write(manifest_path, {'kind': 'real subsets; smoke only, not paper results',
                         'identity': identity, 'selected_ids': selected_ids, 'files_sha256': files})
    return destination


def verify_run(job):
    """Require real optimizer updates, finite saved weights and complete scored outputs."""
    import torch
    from safetensors import safe_open
    from rebuttal.evaluate import validate_records
    checkpoint = Path(json.loads((job.root / 'final_checkpoint.json').read_text())['path'])
    weights = list(checkpoint.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError(f'Smoke checkpoint has no weights: {checkpoint}')
    for path in weights:
        with safe_open(path, framework='pt', device='cpu') as tensors:
            for name in tensors.keys():
                if not torch.isfinite(tensors.get_tensor(name)).all():
                    raise ValueError(f'Non-finite smoke checkpoint tensor: {path}/{name}')
    if job.config['method'] == 'cored':
        pipeline = job.root / 'artifacts/cored'
        for stage in ('phase1', 'distill'):
            counters = [json.loads(p.read_text()) for p in (pipeline / stage).glob('timing-rank-*.json')]
            if ({r['rank'] for r in counters} != set(range(4)) or len(counters) != 4
                    or any(r['optimizer_steps'] < STEPS for r in counters)):
                raise ValueError(f'Smoke did not train all four CoReD ranks in {stage}')
    else:
        cost = json.loads((job.root / 'training_cost.json').read_text())
        if not any(s['stage'] == 'optimization' and s['optimization_steps'] >= STEPS for s in cost['stages']):
            raise ValueError('Smoke did not complete its optimizer updates')
    for name in job.config['evaluation']['math'] + job.config['evaluation']['ood']:
        scored = rows(job.root / 'evaluation' / f'{name}.scored.jsonl')
        validate_records(scored, rows(job.data / 'benchmarks' / f'{name}.jsonl'), name,
                         scored=True, config=job.config)
