import json
import os
from copy import copy
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import get_args
from data_utils.prompt_datasets import PromptDataset
from evaluate_layer_changes import collate_responses
from finetune import get_velocity_field
from utils import get_distillation_schedule, get_model, get_tokenizer, initialize


def one_step_errors(projected, target, velocity_field, layer_index):
    times = projected.new_zeros((projected.size(0),))
    layers = torch.full_like(times, layer_index, dtype=torch.long)
    approximation = projected + velocity_field(projected, times, layers)
    squared_error = (approximation - target).square().sum(dtype=torch.float64)
    cosine_sum = F.cosine_similarity(approximation, target, dim=-1).sum(dtype=torch.float64)
    return torch.stack((squared_error, cosine_sum))


def main():
    args = get_args()
    if not all((args.model_path, args.teacher_model_path, args.data_dir, args.save)):
        raise ValueError("Provide the student checkpoint, teacher checkpoint, data directory, and --save")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("One-step evaluation requires one process")

    checkpoint = Path(args.model_path)
    with (checkpoint / "args.json").open() as f:
        training_args = json.load(f)
    with (checkpoint / "velocity_checkpoint.json").open() as f:
        velocity_checkpoint = json.load(f)
    if training_args["type"] != "rkl-contra-fdd" or velocity_checkpoint["train_seed"] != args.seed:
        raise ValueError("Use the FDD + CONTRA checkpoint and velocity field for the requested training seed")
    args.velocity_field_path = str(checkpoint / "velocity_field.pth")
    args.projector_path = str(checkpoint / "projector.pth")
    args.velocity_d_model = training_args["velocity_d_model"]
    args.velocity_n_layers = training_args["velocity_n_layers"]
    args.num_distill_layers = training_args["num_distill_layers"]

    initialize(args)
    device = torch.cuda.current_device()
    tokenizer = get_tokenizer(args)
    dataset = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)
    dataloader = DataLoader(
        dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers,
        collate_fn=partial(collate_responses, pad_token_id=tokenizer.pad_token_id),
    )
    student = get_model(args, tokenizer, device).transformer.eval()
    teacher_args = copy(args)
    teacher_args.model_path = args.teacher_model_path
    teacher = get_model(teacher_args, tokenizer, device).transformer.eval()
    args.d_student = student.config.n_embd
    args.d_teacher = teacher.config.n_embd
    velocity_field, projector = get_velocity_field(args, device)
    teacher_layers, student_layers = get_distillation_schedule(
        teacher.config.n_layer, student.config.n_layer, args.num_distill_layers,
    )

    errors = torch.zeros((len(student_layers), 2), dtype=torch.float64, device=device)
    num_tokens = 0
    with torch.inference_mode():
        for input_ids, attention_mask, response_mask in tqdm(dataloader, desc=f"Evaluating {args.data_names}"):
            batch_tokens = int(response_mask.sum())
            if not batch_tokens:
                continue
            num_tokens += batch_tokens
            inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            response_mask = response_mask.to(device)
            student_states = student(**inputs, output_hidden_states=True, use_cache=False, return_dict=True).hidden_states
            teacher_states = teacher(**inputs, output_hidden_states=True, use_cache=False, return_dict=True).hidden_states
            for j, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
                projected = projector(student_states[student_layer][response_mask].float())
                target = teacher_states[teacher_layer][response_mask].float()
                errors[j] += one_step_errors(projected, target, velocity_field, j)
    if not num_tokens:
        raise ValueError(f"No reference-response tokens in {args.data_names}")

    mse = errors[:, 0] / (num_tokens * teacher.config.n_embd)
    cosine = errors[:, 1] / num_tokens
    mean_cosine = cosine.mean().item()
    result = {
        "dataset": args.data_names,
        "split": "valid",
        "train_seed": args.seed,
        "num_examples": len(dataset),
        "num_response_tokens": num_tokens,
        "teacher_hidden_size": teacher.config.n_embd,
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,
        "checkpoint_paths": {
            "teacher": args.teacher_model_path,
            "contra_fdd": args.model_path,
            "projector": args.projector_path,
            "velocity_field": args.velocity_field_path,
        },
        "velocity_checkpoint": velocity_checkpoint,
        "velocity_time": 0,
        "step_size": 1,
        "mean_mse": mse.mean().item(),
        "mean_cosine_similarity": mean_cosine,
        "mean_cosine_distance": 1 - mean_cosine,
        "layers": [
            {
                "student_layer": int(student_layer),
                "teacher_layer": int(teacher_layer),
                "mse": layer_mse,
                "cosine_similarity": layer_cosine,
                "cosine_distance": 1 - layer_cosine,
            }
            for student_layer, teacher_layer, layer_mse, layer_cosine
            in zip(student_layers, teacher_layers, mse.tolist(), cosine.tolist())
        ],
    }
    dist.destroy_process_group()
    path = Path(args.save) / "one_step.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    temporary.replace(path)
    print(f"Saved one-step evaluation to {path}")


if __name__ == "__main__":
    main()
