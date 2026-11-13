#!/usr/bin/env python3
"""Measure GPT-2 hidden-state CKA for uniform and fixed non-uniform pairings.

Calibration and CKA holdout use disjoint records from Dolly's training split.
Neither selection nor diagnostics use the downstream evaluation benchmarks.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from layer_change_metrics import encode_example


def nonuniform_depth_schedule(teacher_depth, count, uniform_teacher, exponent=1.5):
    """Fixed depth warp; independent of CKA scores and downstream results."""
    schedule = [round(teacher_depth * ((j + 1) / (count + 1)) ** exponent) for j in range(count)]
    if (any(x < 1 or x >= teacher_depth for x in schedule)
            or any(a >= b for a, b in zip(schedule, schedule[1:]))
            or schedule == uniform_teacher):
        raise ValueError("Depth-warp schedule is not a distinct ordered interior pairing")
    return schedule


def cka_selected_schedule(scores, uniform_teacher, student_layers, max_shift=8):
    """Choose an ordered teacher schedule by calibration CKA only."""
    states = {0: (0.0, ())}
    teacher_depth = len(scores) - 1
    for j, student in enumerate(student_layers):
        next_states = {}
        low = max(1, uniform_teacher[j] - max_shift)
        high = min(teacher_depth - 1, uniform_teacher[j] + max_shift)
        for previous, (total, path) in states.items():
            for teacher in range(max(low, previous + 1), high + 1):
                candidate = (total + scores[teacher][student], path + (teacher,))
                if teacher not in next_states or candidate[0] > next_states[teacher][0]:
                    next_states[teacher] = candidate
        states = next_states
    if not states:
        raise ValueError("No ordered CKA pairing under max_shift")
    return list(max(states.values(), key=lambda value: value[0])[1])


def collect_grams(model, examples, num_layers, torch, device):
    features = [[] for _ in range(num_layers + 1)]
    model.eval()
    with torch.inference_mode():
        for input_ids, positions in examples:
            ids = torch.tensor([input_ids], device=device)
            outputs = model.base_model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
            if len(outputs.hidden_states) != num_layers + 1:
                raise ValueError("Unexpected number of hidden states")
            for layer, hidden in enumerate(outputs.hidden_states):
                features[layer].append(hidden[0, positions].float().cpu())
            del outputs
    grams = []
    for pieces in features:
        x = torch.cat(pieces).to(device=device, dtype=torch.float32)
        x -= x.mean(dim=0, keepdim=True)
        gram = x @ x.T
        grams.append(gram.cpu())
    return torch.stack(grams)


def cka_matrix(teacher_grams, student_grams, torch):
    t = teacher_grams.flatten(1).double()
    s = student_grams.flatten(1).double()
    numerator = t @ s.T
    denominator = t.norm(dim=1)[:, None] * s.norm(dim=1)[None, :]
    return (numerator / denominator.clamp_min(1e-12)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--data", required=True, help="Raw Dolly training JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-distill-layers", type=int, default=6)
    parser.add_argument("--calibration-examples", type=int, default=64)
    parser.add_argument("--holdout-examples", type=int, default=64)
    parser.add_argument("--max-tokens-per-example", type=int, default=8)
    parser.add_argument("--depth-exponent", type=float, default=1.5,
                        help="Fixed non-uniform teacher-depth warp; never fitted to CKA")
    parser.add_argument("--max-cka-shift", type=int, default=8,
                        help="Maximum teacher-layer shift from uniform for CKA-selected pairing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if min(args.calibration_examples, args.holdout_examples, args.max_tokens_per_example) <= 0:
        parser.error("Example and token counts must be positive")
    if args.depth_exponent <= 0:
        parser.error("depth-exponent must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    teacher_path, student_path = Path(args.teacher), Path(args.student)
    teacher_cfg = json.loads((teacher_path / "config.json").read_text())
    student_cfg = json.loads((student_path / "config.json").read_text())
    teacher_depth, student_depth = teacher_cfg["n_layer"], student_cfg["n_layer"]
    if teacher_cfg["vocab_size"] != student_cfg["vocab_size"]:
        raise ValueError("Teacher and student must share token IDs")
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    rng = random.Random(args.seed)
    examples = []
    with open(args.data, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream):
            if line_no < 1000:  # Match prepare_offline_dolly.py's train split.
                continue
            row = json.loads(line)
            ids, positions = encode_example(tokenizer, row, 512, 256)
            positions = sorted(rng.sample(positions, min(len(positions), args.max_tokens_per_example)))
            examples.append((ids, positions))
            if len(examples) == args.calibration_examples + args.holdout_examples:
                break
    if len(examples) != args.calibration_examples + args.holdout_examples:
        raise ValueError("Not enough Dolly training records for calibration and holdout")

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.float16, local_files_only=True, attn_implementation="eager"
    ).eval().to("cuda:0")
    t_cal = collect_grams(teacher, examples[:args.calibration_examples], teacher_depth, torch, "cuda:0")
    t_hold = collect_grams(teacher, examples[args.calibration_examples:], teacher_depth, torch, "cuda:0")
    del teacher
    torch.cuda.empty_cache()
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.float16, local_files_only=True, attn_implementation="eager"
    ).eval().to("cuda:0")
    s_cal = collect_grams(student, examples[:args.calibration_examples], student_depth, torch, "cuda:0")
    s_hold = collect_grams(student, examples[args.calibration_examples:], student_depth, torch, "cuda:0")
    del student
    cal = cka_matrix(t_cal, s_cal, torch)
    hold = cka_matrix(t_hold, s_hold, torch)
    teacher_uniform = [int(x) for x in np.linspace(0, teacher_depth, args.num_distill_layers + 2)[1:-1]]
    student_uniform = [int(x) for x in np.linspace(0, student_depth, args.num_distill_layers + 2)[1:-1]]
    teacher_nonuniform = nonuniform_depth_schedule(
        teacher_depth, args.num_distill_layers, teacher_uniform, args.depth_exponent
    )
    teacher_cka = cka_selected_schedule(cal, teacher_uniform, student_uniform, args.max_cka_shift)

    def pairing_scores(matrix, teacher_indices):
        return [matrix[t][s] for t, s in zip(teacher_indices, student_uniform)]

    for name, matrix in [("calibration", cal), ("holdout", hold)]:
        with (output / f"cka_{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["teacher\\student", *range(student_depth + 1)])
            writer.writerows([[index, *row] for index, row in enumerate(matrix)])
    schedule = {
        "num_teacher_layers": teacher_depth, "num_student_layers": student_depth,
        "teacher": teacher_nonuniform, "student": student_uniform,
    }
    (output / "nonuniform_pairing.json").write_text(json.dumps(schedule, indent=2) + "\n")
    cka_schedule = {**schedule, "teacher": teacher_cka}
    (output / "cka_pairing.json").write_text(json.dumps(cka_schedule, indent=2) + "\n")
    summary = {
        "calibration": {"teacher": teacher_uniform, "student": student_uniform,
                        "uniform_cka": pairing_scores(cal, teacher_uniform),
                        "nonuniform_cka": pairing_scores(cal, teacher_nonuniform),
                        "cka_selected_cka": pairing_scores(cal, teacher_cka)},
        "holdout": {"uniform_cka": pairing_scores(hold, teacher_uniform),
                    "nonuniform_cka": pairing_scores(hold, teacher_nonuniform),
                    "cka_selected_cka": pairing_scores(hold, teacher_cka)},
        "nonuniform_teacher": teacher_nonuniform,
        "cka_selected_teacher": teacher_cka,
        "nonuniform_rule": f"round(teacher_depth * depth_fraction ** {args.depth_exponent})",
        "data": args.data, "calibration_examples": args.calibration_examples,
        "holdout_examples": args.holdout_examples, "seed": args.seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Layer-pair CKA", "", "CKA uses raw hidden states on disjoint Dolly training records.",
             "CKA-selected pairing maximizes calibration CKA under ordered layers and a shift limit.",
             "Fixed non-uniform pairing is a separate depth-warp control independent of CKA.",
             "Holdout records do not participate in CKA pairing selection.", "",
             "| Pair | Student | Uniform teacher | Uniform CKA (holdout) | CKA-selected teacher | Selected CKA (holdout) | Fixed non-uniform teacher | Fixed CKA (holdout) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for j, student_layer in enumerate(student_uniform):
        lines.append(f"| {j+1} | {student_layer} | {teacher_uniform[j]} | {summary['holdout']['uniform_cka'][j]:.4f} | {teacher_cka[j]} | {summary['holdout']['cka_selected_cka'][j]:.4f} | {teacher_nonuniform[j]} | {summary['holdout']['nonuniform_cka'][j]:.4f} |")
    for label, key in [("Uniform", "uniform_cka"), ("CKA-selected", "cka_selected_cka"), ("Fixed non-uniform", "nonuniform_cka")]:
        vals = summary["holdout"][key]
        lines.append(f"\n{label} mean holdout CKA: {sum(vals)/len(vals):.4f}.")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"CKA report: {output / 'report.md'}")
    print(f"CKA-selected pairing: {output / 'cka_pairing.json'}")
    print(f"Non-uniform pairing: {output / 'nonuniform_pairing.json'}")


if __name__ == "__main__":
    main()
