import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--baseline-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Student training seed")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--num-calibration", type=int, default=128)
    parser.add_argument("--num-eval", type=int, default=256)
    parser.add_argument("--tokens-per-example", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.num_calibration, args.num_eval) < 2:
        parser.error("Calibration and evaluation each require at least two sequences")
    if min(args.eval_batch_size, args.tokens_per_example, args.max_prompt_length,
           args.bootstrap_samples) < 1 or args.max_length <= args.max_prompt_length:
        parser.error("Use positive sizes and --max-length greater than --max-prompt-length")
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--ridge must be positive and finite")
    return args


def load_examples(args, tokenizer):
    examples, seen = [], set()
    with args.data_path.open() as f:
        for index, line in enumerate(f):
            row = json.loads(line)
            prompt = row["prompt"].replace("<n>", "\n")
            response = row["output"]
            if isinstance(response, list):
                response = response[0]
            prompt_ids = tokenizer.encode(prompt)[:args.max_prompt_length]
            response_ids = tokenizer.encode(response)[:args.max_length - args.max_prompt_length]
            if not prompt_ids or not response_ids or tuple(prompt_ids) in seen:
                continue
            seen.add(tuple(prompt_ids))
            examples.append((index, prompt_ids, response_ids))
    np.random.default_rng(args.sample_seed).shuffle(examples)
    needed = args.num_calibration + args.num_eval
    if len(examples) < needed:
        raise ValueError(f"Need {needed} distinct usable prompts in {args.data_path}, found {len(examples)}")
    return examples[:args.num_calibration], examples[args.num_calibration:needed]


def collate(samples, pad_token_id, tokens_per_example, calibration):
    length = max(len(prompt) + len(response) for _, prompt, response in samples)
    input_ids = torch.full((len(samples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    sample_mask = torch.zeros_like(attention_mask)
    for i, (_, prompt, response) in enumerate(samples):
        end = len(prompt) + len(response)
        input_ids[i, :end] = torch.tensor(prompt + response)
        attention_mask[i, :end] = True
        start = 0 if calibration else len(prompt)
        # One token at the center of each equally spaced bin.
        count = min(tokens_per_example, end - start)
        positions = start + (2 * torch.arange(count) + 1) * (end - start) // (2 * count)
        sample_mask[i, positions] = True
    return {"input_ids": input_ids, "attention_mask": attention_mask}, sample_mask


def forward_states(model, inputs):
    block_inputs = {}
    last_residual = []

    def capture_block(module, args, kwargs):
        block_inputs["args"] = args[1:]
        block_inputs["kwargs"] = kwargs

    def capture_norm(module, args):
        last_residual.append(args[0])

    handles = [
        model.transformer.h[0].register_forward_pre_hook(capture_block, with_kwargs=True),
        model.transformer.ln_f.register_forward_pre_hook(capture_norm),
    ]
    try:
        outputs = model.transformer(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    # Transformers exposes the last state after ln_f. Use the pre-ln_f state at every boundary.
    return (*outputs.hidden_states[:-1], last_residual[0]), block_inputs


def run_block(block, state, block_inputs):
    result = block(state, *block_inputs["args"], **block_inputs["kwargs"])
    return result[0] if isinstance(result, tuple) else result


def fit_ridge(source, target, strength):
    source_mean, target_mean = source.mean(dim=0), target.mean(dim=0)
    x, y = source - source_mean, target - target_mean
    covariance = x.T @ x
    covariance.diagonal().add_(strength * covariance.diagonal().mean().clamp_min(1e-8))
    weight = torch.linalg.solve(covariance, x.T @ y)
    return weight, target_mean - source_mean @ weight


def calibrate(teacher, students, loader, boundaries, device, strength):
    teacher_rows = []
    student_rows = {name: [] for name in students}
    for inputs, mask in tqdm(loader, desc="Calibrating adapters"):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        mask = mask.to(device)
        states, _ = forward_states(teacher, inputs)
        teacher_rows.append(torch.cat([states[j][mask].float() for j in boundaries]))
        for name, model in students.items():
            states, _ = forward_states(model, inputs)
            student_rows[name].append(torch.cat([state[mask].float() for state in states]))
    target = torch.cat(teacher_rows)
    adapters = {}
    for name, batches in student_rows.items():
        source = torch.cat(batches)
        project_weight, project_bias = fit_ridge(source, target, strength)
        lift_weight, lift_bias = fit_ridge(target, source, strength)
        adapters[name] = (project_weight, project_bias, lift_weight, lift_bias)
    return adapters


def centered_grams(states):
    centered = states - states.mean(dim=-2, keepdim=True)
    gram = centered @ centered.transpose(-1, -2)
    return F.normalize(gram.flatten(start_dim=-2), dim=-1)


def linear_cka(left, right):
    return (centered_grams(left) * centered_grams(right)).sum().item()


def projected_update_metrics(predicted, target):
    return {
        "cosine": F.cosine_similarity(predicted, target, dim=-1).mean().item(),
        "normalized_mse": (
            (predicted - target).square().sum() / target.square().sum().clamp_min(1e-8)
        ).item(),
    }


def teacher_log_probs(teacher, state, mask):
    return F.log_softmax(teacher.lm_head(teacher.transformer.ln_f(state)[mask]), dim=-1, dtype=torch.float32)


def score_replacement(teacher, candidate, teacher_states, start, end, block_inputs, mask, reference):
    target_update = (teacher_states[end][mask].float() - teacher_states[start][mask].float())
    errors = (candidate[mask].float() - teacher_states[end][mask].float()).square().sum(dim=-1)
    energy = target_update.square().sum(dim=-1)
    for block in teacher.transformer.h[end:]:
        candidate = run_block(block, candidate, block_inputs)
    candidate_log_probs = teacher_log_probs(teacher, candidate, mask)
    kl = (reference.exp() * (reference - candidate_log_probs)).sum(dim=-1).clamp_min(0)
    row_ids = mask.nonzero(as_tuple=True)[0]
    totals = errors.new_zeros((mask.size(0), 3))
    totals.index_add_(0, row_ids, torch.stack((errors, energy, kl), dim=-1))
    return torch.stack((
        totals[:, 0] / totals[:, 1].clamp_min(1e-8),
        totals[:, 2] / mask.sum(dim=1),
    ), dim=-1).cpu().numpy()


def replace_interval(model, student_inputs, teacher_start, block, adapter):
    project_weight, _, lift_weight, lift_bias = adapter
    lifted = (teacher_start.float() @ lift_weight + lift_bias).to(model.dtype)
    output = run_block(model.transformer.h[block], lifted, student_inputs)
    update = (output.float() - lifted.float()) @ project_weight
    return (teacher_start.float() + update).to(teacher_start.dtype)


def paired_interval(values, resamples):
    means = values[resamples].mean(axis=1)
    return {
        name: {"mean": float(values[:, j].mean()), "ci95": np.quantile(means[:, j], [0.025, 0.975]).tolist()}
        for j, name in enumerate(("normalized_mse", "teacher_kl"))
    }


def study_schedule(path, teacher_depth, student_depth, train_seed):
    training_args_path = path / "args.json"
    if training_args_path.is_file():
        with training_args_path.open() as f:
            args = json.load(f)
        if args["seed"] != train_seed:
            raise ValueError(f"Checkpoint {path} has training seed {args['seed']}, expected {train_seed}")
        count = args["num_distill_layers"]
    else:
        count = 6
    # Matches get_distillation_schedule in utils.py.
    teacher_layers = np.linspace(0, teacher_depth, count + 2, dtype=int)[1:-1].tolist()
    student_layers = np.linspace(0, student_depth, count + 2, dtype=int)[1:-1].tolist()
    return list(zip(student_layers, teacher_layers))


def get_saved_projector(path, teacher, student, seed, device):
    metadata_path, projector_path = path / "velocity_checkpoint.json", path / "projector.pth"
    if not (metadata_path.is_file() and projector_path.is_file()):
        return None
    with metadata_path.open() as f:
        metadata = json.load(f)
    if metadata["train_seed"] != seed:
        raise ValueError(f"Saved projector in {path} belongs to seed {metadata['train_seed']}")
    from distillm.projector import Projector

    projector = Projector(student.config.n_embd, teacher.config.n_embd).to(device).eval()
    projector.load_state_dict(torch.load(projector_path, map_location="cpu", weights_only=True))
    return projector


def summarize_replacements(scores, student_names, resamples):
    contrasts = {"contra_fdd_minus_fdd": scores["contra_fdd"] - scores["fdd"]}
    for name in student_names:
        for control in ("skip_all", "keep_last", f"{name}_neighbor"):
            contrasts[f"{name}_minus_{control}"] = scores[name] - scores[control]
    return {
        "candidates": {name: paired_interval(value, resamples) for name, value in scores.items()},
        "paired_differences": {name: paired_interval(value, resamples) for name, value in contrasts.items()},
    }


def representation_scores(teacher, student, skipped, boundaries, schedule, adapter, projector):
    project_weight, project_bias, _, _ = adapter
    projected = student @ project_weight + project_bias
    saved_projection = projector(student) if projector is not None else None
    layer_cka = (centered_grams(student) @ centered_grams(teacher).T).tolist()
    teacher_updates = teacher[boundaries[1:]] - teacher[boundaries[:-1]]
    student_updates = student[1:] - student[:-1]
    update_cka = (centered_grams(student_updates) @ centered_grams(teacher_updates).T).tolist()
    anchor_pairs = set(schedule)
    blocks = []
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        target = teacher_updates[i]
        projected_delta = projected[i + 1] - projected[i]
        blocks.append({
            "student_block": i + 1, "teacher_start": start, "teacher_end": end,
            "start_is_training_anchor": (i, start) in anchor_pairs,
            "end_is_training_anchor": (i + 1, end) in anchor_pairs,
            "state_cka": {
                "endpoint": layer_cka[i + 1][end],
                "pooled": linear_cka(student[i + 1], teacher[start + 1:end + 1].mean(dim=0)),
                "keep_last": linear_cka(student[i + 1], skipped[i]),
                "skip_all": layer_cka[i + 1][start],
            },
            "update_cka": {
                "full_interval": update_cka[i][i],
                "last_natural_update": linear_cka(student_updates[i], teacher[end] - teacher[end - 1]),
                "keep_last": linear_cka(student_updates[i], skipped[i] - teacher[start]),
            },
            "ridge_update": projected_update_metrics(projected_delta, target),
            "saved_projector_update": projected_update_metrics(
                saved_projection[i + 1] - saved_projection[i], target,
            ) if saved_projection is not None else None,
        })
    anchors = [
        {"student_layer": s, "teacher_layer": t, "state_cka": layer_cka[s][t]}
        for s, t in schedule
    ]
    transitions = []
    for (s0, t0), (s1, t1) in zip(schedule, schedule[1:]):
        target = teacher[t1] - teacher[t0]
        transitions.append({
            "student_layers": [s0, s1], "teacher_layers": [t0, t1],
            "update_cka": linear_cka(student[s1] - student[s0], target),
            "ridge_update": projected_update_metrics(projected[s1] - projected[s0], target),
            "saved_projector_update": projected_update_metrics(
                saved_projection[s1] - saved_projection[s0], target,
            ) if saved_projection is not None else None,
        })
    return {
        "state_cka_student_by_teacher": layer_cka,
        "update_cka_student_by_teacher_interval": update_cka,
        "blocks": blocks, "training_anchors": anchors, "training_anchor_transitions": transitions,
    }


@torch.inference_mode()
def main():
    args = get_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    paths = {"teacher": args.teacher_model_path, "fdd": args.baseline_model_path, "contra_fdd": args.model_path}
    models = {
        name: AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=dtype).to(device).eval()
        for name, path in paths.items()
    }
    teacher = models.pop("teacher")
    student_depth = models["fdd"].config.n_layer
    teacher_depth = teacher.config.n_layer
    if student_depth < 2 or teacher_depth <= student_depth or teacher_depth % student_depth:
        raise ValueError("Expected at least two student blocks and an integer teacher-to-student depth ratio > 1")
    for model in [teacher, *models.values()]:
        if model.config.model_type != "gpt2" or model.config.n_positions < args.max_length:
            raise ValueError("Expected GPT-2 checkpoints with enough positions for --max-length")
    if models["contra_fdd"].config.n_layer != student_depth:
        raise ValueError("Student checkpoint depths differ")
    boundaries = list(range(0, teacher_depth + 1, teacher_depth // student_depth))
    schedules = {
        name: study_schedule(paths[name], teacher_depth, student_depth, args.seed)
        for name in models
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    calibration, evaluation = load_examples(args, tokenizer)
    calibration_loader = DataLoader(
        calibration, batch_size=args.eval_batch_size,
        collate_fn=partial(collate, pad_token_id=tokenizer.eos_token_id,
                           tokens_per_example=max(16, args.tokens_per_example), calibration=True),
    )
    evaluation_loader = DataLoader(
        evaluation, batch_size=args.eval_batch_size,
        collate_fn=partial(collate, pad_token_id=tokenizer.eos_token_id,
                           tokens_per_example=args.tokens_per_example, calibration=False),
    )
    adapters = calibrate(teacher, models, calibration_loader, boundaries, device, args.ridge)
    projector = get_saved_projector(args.model_path, teacher, models["contra_fdd"], args.seed, device)
    states = {name: [] for name in ("teacher", *models, "keep_last")}
    replacements = [{} for _ in range(student_depth)]

    for inputs, mask in tqdm(evaluation_loader, desc=f"Evaluating seed {args.seed}"):
        inputs = {key: value.to(device) for key, value in inputs.items()}
        mask = mask.to(device)
        teacher_states, teacher_inputs = forward_states(teacher, inputs)
        states["teacher"].append(torch.stack([state[mask].float() for state in teacher_states]))
        reference = teacher_log_probs(teacher, teacher_states[-1], mask)
        student_inputs = {}
        for name, model in models.items():
            student_states, student_inputs[name] = forward_states(model, inputs)
            states[name].append(torch.stack([state[mask].float() for state in student_states]))
        skipped_states = []
        for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            z = teacher_states[start]
            keep_last = run_block(teacher.transformer.h[end - 1], z, teacher_inputs)
            skipped_states.append(keep_last[mask].float())
            candidates = {"skip_all": z, "keep_last": keep_last}
            neighbor = i + 1 if i + 1 < student_depth else i - 1
            for name, model in models.items():
                candidates[name] = replace_interval(model, student_inputs[name], z, i, adapters[name])
                candidates[f"{name}_neighbor"] = replace_interval(
                    model, student_inputs[name], z, neighbor, adapters[name],
                )
            for name, candidate in candidates.items():
                score = score_replacement(
                    teacher, candidate, teacher_states, start, end, teacher_inputs, mask, reference,
                )
                replacements[i].setdefault(name, []).append(score)
        states["keep_last"].append(torch.stack(skipped_states))

    states = {name: torch.cat(batches, dim=1) for name, batches in states.items()}
    representation = {
        name: representation_scores(
            states["teacher"], states[name], states["keep_last"], boundaries, schedules[name],
            adapters[name], projector if name == "contra_fdd" else None,
        )
        for name in models
    }
    resamples = np.random.default_rng(args.sample_seed).integers(
        len(evaluation), size=(args.bootstrap_samples, len(evaluation)),
    )
    scores_by_block = [{name: np.concatenate(values) for name, values in block.items()} for block in replacements]
    mean_scores = {
        name: np.mean([block[name] for block in scores_by_block], axis=0) for name in scores_by_block[0]
    }
    replacement_scores = [
        {
            "student_block": i + 1,
            "neighbor_block": i + 2 if i + 1 < student_depth else i,
            "teacher_start": boundaries[i],
            "teacher_end": boundaries[i + 1],
            **summarize_replacements(scores, models, resamples),
        }
        for i, scores in enumerate(scores_by_block)
    ]

    result = {
        "train_seed": args.seed,
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "saved_projector": str(args.model_path / "projector.pth") if projector is not None else None,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "calibration_record_indices": [example[0] for example in calibration],
        "evaluation_record_indices": [example[0] for example in evaluation],
        "num_evaluation_tokens": states["teacher"].size(1),
        "teacher_boundaries": boundaries,
        "representation": representation,
        "mean_block_replacement": summarize_replacements(mean_scores, models, resamples),
        "block_replacement": replacement_scores,
    }
    args.save.mkdir(parents=True, exist_ok=True)
    path = args.save / "coarse_dynamics.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    temporary.replace(path)
    print(f"Saved coarse-dynamics evaluation to {path}")


if __name__ == "__main__":
    main()
