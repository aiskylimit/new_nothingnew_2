import json
import os
from copy import copy
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import get_args
from data_utils.prompt_datasets import PromptDataset
from distillm.projector import Projector
from utils import get_distillation_schedule, get_model, get_tokenizer, initialize


def collate_responses(samples, pad_token_id):
    length = max(len(prompt) + len(response) for _, prompt, response in samples)
    input_ids = torch.full((len(samples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(attention_mask)
    for i, (_, prompt, response) in enumerate(samples):
        end = len(prompt) + len(response)
        input_ids[i, :end] = torch.tensor(prompt + response, dtype=torch.long)
        attention_mask[i, :end] = True
        response_mask[i, len(prompt):end] = True
    return input_ids, attention_mask, response_mask


def layer_changes(model, inputs, response_mask, schedule, projector=None):
    states = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True).hidden_states
    selected = [states[layer][response_mask].float() for layer in schedule]
    if projector is not None:
        selected = [projector(state) for state in selected]
    return [end - start for start, end in zip(selected, selected[1:])]


def main():
    args = get_args()
    if not all((args.model_path, args.baseline_model_path, args.teacher_model_path, args.projector_path, args.save)):
        raise ValueError("Provide both student checkpoints, the teacher, the CONTRA projector, and --save")
    if args.num_distill_layers < 2 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Layer-change evaluation requires at least two layer positions and one process")
    initialize(args)
    device = torch.cuda.current_device()
    tokenizer = get_tokenizer(args)
    dataset = PromptDataset(args, tokenizer, "valid", args.data_dir, args.dev_num)
    dataloader = DataLoader(
        dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers,
        collate_fn=partial(collate_responses, pad_token_id=tokenizer.pad_token_id),
    )

    contra = get_model(args, tokenizer, device).transformer.eval()
    model_args = copy(args)
    model_args.model_path = args.baseline_model_path
    fdd = get_model(model_args, tokenizer, device).transformer.eval()
    model_args.model_path = args.teacher_model_path
    teacher = get_model(model_args, tokenizer, device).transformer.eval()
    if (fdd.config.n_layer, fdd.config.n_embd) != (contra.config.n_layer, contra.config.n_embd):
        raise ValueError("The student layer counts and hidden dimensions must match")
    teacher_layers, student_layers = get_distillation_schedule(
        teacher.config.n_layer, contra.config.n_layer, args.num_distill_layers,
    )
    projector_path = Path(args.projector_path)
    with (projector_path.parent / "velocity_checkpoint.json").open() as f:
        projector_checkpoint = json.load(f)
    if projector_checkpoint["train_seed"] != args.seed:
        raise ValueError("The CONTRA projector must come from the requested training seed")
    projector = Projector(contra.config.n_embd, teacher.config.n_embd).to(device).eval()
    projector.load_state_dict(torch.load(projector_path, map_location="cpu", weights_only=True))

    models = {"fdd": fdd, "contra_fdd": contra}
    errors = torch.zeros((len(models), len(student_layers) - 1), dtype=torch.float64, device=device)
    num_tokens = 0
    with torch.inference_mode():
        for input_ids, attention_mask, response_mask in tqdm(dataloader, desc=f"Evaluating {args.data_names}"):
            num_tokens += int(response_mask.sum())
            inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
            response_mask = response_mask.to(device)
            target = layer_changes(teacher, inputs, response_mask, teacher_layers)
            for i, model in enumerate(models.values()):
                predicted = layer_changes(model, inputs, response_mask, student_layers, projector)
                errors[i] += torch.stack([
                    (student_delta - teacher_delta).square().sum(dtype=torch.float64)
                    for student_delta, teacher_delta in zip(predicted, target)
                ])
    if num_tokens == 0:
        raise ValueError(f"No reference-response tokens in {args.data_names}")
    mse = errors / (num_tokens * teacher.config.n_embd)
    fdd_mse, contra_mse = mse.tolist()
    paired_diff = (mse[1] - mse[0]).tolist()
    result = {
        "dataset": args.data_names,
        "train_seed": args.seed,
        "num_examples": len(dataset),
        "num_response_tokens": num_tokens,
        "teacher_hidden_size": teacher.config.n_embd,
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,
        "checkpoint_paths": {
            "teacher": args.teacher_model_path,
            "fdd": args.baseline_model_path,
            "contra_fdd": args.model_path,
        },
        "projector_path": str(projector_path),
        "projector_checkpoint": projector_checkpoint,
        "mean_transition_mse": {
            "fdd": mse[0].mean().item(),
            "contra_fdd": mse[1].mean().item(),
            "contra_fdd_minus_fdd": (mse[1] - mse[0]).mean().item(),
        },
        "transitions": [
            {
                "student_layers": student_layers[i:i + 2].tolist(),
                "teacher_layers": teacher_layers[i:i + 2].tolist(),
                "fdd_mse": fdd_mse[i],
                "contra_fdd_mse": contra_mse[i],
                "contra_fdd_minus_fdd": paired_diff[i],
            }
            for i in range(len(fdd_mse))
        ],
    }
    dist.destroy_process_group()
    path = Path(args.save) / "layer_changes.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    temporary.replace(path)


if __name__ == "__main__":
    main()
