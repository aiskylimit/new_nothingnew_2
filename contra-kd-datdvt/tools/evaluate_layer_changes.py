#!/usr/bin/env python3
"""Offline GPT-2 checkpoint diagnostic using the repository's FDD readout.

No fitting, generation, gradient updates, W&B, or model/data downloads. Runs
one example at a time and chunks vocabulary projections to bound memory.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

from layer_change_metrics import (
    aggregate, encode_example, paired_bootstrap, uniform_schedule,
)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def named_paths(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        if name in result:
            raise ValueError(f"Duplicate name {name!r}; use a distinct name")
        result[name] = Path(path).expanduser().resolve()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--data", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--compare", action="append", default=[], metavar="BASELINE=AUGMENTED")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--max-prompt-length", type=positive_int, default=256)
    parser.add_argument("--num-distill-layers", type=positive_int, default=6)
    parser.add_argument("--allow-duplicate-layer-schedule", action="store_true",
                        help="Reproduce duplicated indices from utils.get_distillation_schedule when its total is too small")
    parser.add_argument("--token-chunk-size", type=positive_int, default=16)
    parser.add_argument("--bootstrap-samples", type=positive_int, default=2000)
    parser.add_argument("--max-examples", type=positive_int, help="First N distinct records; diagnostic pilot only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--zero-norm-eps", type=float, default=1e-5)
    args = parser.parse_args()
    if args.max_prompt_length >= args.max_length - 1:
        parser.error("max-prompt-length must leave room for a response target")
    if args.zero_norm_eps <= 0:
        parser.error("zero-norm-eps must be positive")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_metadata(path):
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing local checkpoint: {path}. Provision download.txt first.")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gpt2":
        raise ValueError(f"This diagnostic supports full GPT-2 checkpoints only: {path}")
    weights = sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")))
    if not weights:
        raise FileNotFoundError(f"No full model weights found in {path}")
    selected_args = {}
    if (path / "args.json").is_file():
        training = json.loads((path / "args.json").read_text())
        # Whitelist: never copy credentials or an entire training config.
        selected_args = {key: training.get(key) for key in [
            "type", "seed", "epochs", "lr", "batch_size", "gradient_accumulation_steps",
            "num_distill_layers", "num_teacher_layers", "num_student_layers",
        ]}
    return {
        "path": str(path), "config_sha256": file_hash(path / "config.json"),
        "args_sha256": file_hash(path / "args.json") if (path / "args.json").exists() else None,
        "config": {key: config.get(key) for key in ["model_type", "n_layer", "n_embd", "vocab_size"]},
        "training": selected_args,
        "weights": [{"file": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights],
        "provenance_note": "Local weight size/mtime recorded, not a cryptographic weight hash or verified Hub revision.",
    }


def read_examples(path, tokenizer, args):
    examples, seen = [], set()
    duplicate_count = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                input_ids, positions = encode_example(tokenizer, row, args.max_length, args.max_prompt_length)
                identity = json.dumps({key: row.get(key, "") for key in ["instruction", "input", "output"]}, sort_keys=True)
                record_id = hashlib.sha256(identity.encode()).hexdigest()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid evaluation record {path}:{line_number}: {exc}") from exc
            if record_id in seen:
                duplicate_count += 1
                continue
            seen.add(record_id)
            examples.append({"id": record_id, "line": line_number, "input_ids": input_ids, "positions": positions})
            if args.max_examples and len(examples) >= args.max_examples:
                break
    if not examples:
        raise ValueError(f"No evaluation examples in {path}")
    return examples, {
        "path": str(path), "sha256": file_hash(path), "examples": len(examples),
        "duplicate_records_skipped_in_read_prefix": duplicate_count,
        "selection": "first N distinct records" if args.max_examples else "all distinct records",
        "max_examples": args.max_examples,
        "reference_policy": "first reference when output is a list",
        "tokenized_input_sha256": hashlib.sha256(json.dumps([
            [e["id"], e["input_ids"], e["positions"]] for e in examples
        ], separators=(",", ":")).encode()).hexdigest(),
        "split_note": "Caller-provided evaluation split; no automatic training-overlap certification.",
    }


def start_gpu_sampler(output):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        return None, None, {"available": False, "reason": "Need one explicit CUDA_VISIBLE_DEVICES ID"}
    stream = (output / "gpu_samples.csv").open("w")
    stream.write("timestamp,index,uuid,utilization_gpu_percent,memory_used_mib,power_draw_w\n")
    stream.flush()
    try:
        process = subprocess.Popen([
            "nvidia-smi", f"--id={visible}",
            "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits", "--loop-ms=1000",
        ], stdout=stream, stderr=subprocess.DEVNULL)
    except OSError as exc:
        stream.close()
        return None, None, {"available": False, "reason": str(exc)}
    return process, stream, {"available": True, "path": "gpu_samples.csv", "interval_ms": 1000}


def stop_gpu_sampler(process, stream, metadata):
    if process is None:
        return
    if process.poll() is not None:
        metadata.update(available=False, reason=f"nvidia-smi exited with code {process.returncode}")
    else:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    stream.close()


def selected_hiddens(model, input_ids, positions, layers, torch):
    # Calling the transformer trunk avoids allocating unused final-layer logits.
    output = model.base_model(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    selected = [output.hidden_states[layer][0, positions].detach().cpu() for layer in layers]
    del output
    return selected


def project(model, hidden, torch):
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight") or not hasattr(model, "lm_head"):
        raise AttributeError("Expected a GPT-2 causal LM with an exposed lm_head")
    if head is not model.lm_head:
        raise AttributeError("model.lm_head differs from get_output_embeddings(); refusing incompatible FDD readout")
    hidden = hidden.to(device=head.weight.device, dtype=head.weight.dtype)
    # Match get_fdd_loss: direct LM head, no added final LayerNorm or temperature.
    return torch.log_softmax(head(hidden).float(), dim=-1)


def compare_example(teacher, student, example, teacher_layers, student_layers, args, torch):
    inputs = torch.tensor([example["input_ids"]], device="cuda:0", dtype=torch.long)
    positions = example["positions"]
    t_states = selected_hiddens(teacher, inputs, positions, teacher_layers, torch)
    s_states = selected_hiddens(student, inputs, positions, student_layers, torch)
    transitions = [{
        "tokens": 0, "valid_tokens": 0, "cosine_sum": 0.0,
        "delta_error_sq": 0.0, "teacher_delta_sq": 0.0,
    } for _ in range(len(teacher_layers) - 1)]
    state_error_sq, teacher_state_sq = 0.0, 0.0
    for start in range(0, len(positions), args.token_chunk_size):
        previous_t, previous_s = None, None
        for j, (t_hidden, s_hidden) in enumerate(zip(t_states, s_states)):
            t = project(teacher, t_hidden[start:start + args.token_chunk_size], torch)
            s = project(student, s_hidden[start:start + args.token_chunk_size], torch)
            state_error_sq += (s - t).double().square().sum().item()
            teacher_state_sq += t.double().square().sum().item()
            if j:
                dt, ds = t - previous_t, s - previous_s
                tnorm, snorm = dt.norm(dim=-1), ds.norm(dim=-1)
                valid = (tnorm > args.zero_norm_eps) & (snorm > args.zero_norm_eps)
                cosine = (ds[valid] * dt[valid]).sum(-1) / (snorm[valid] * tnorm[valid])
                stats = transitions[j - 1]
                stats["tokens"] += len(t)
                stats["valid_tokens"] += int(valid.sum().item())
                stats["cosine_sum"] += (1 - cosine.clamp(-1, 1)).double().sum().item()
                stats["delta_error_sq"] += (ds - dt).double().square().sum().item()
                stats["teacher_delta_sq"] += dt.double().square().sum().item()
            previous_t, previous_s = t, s
    result = {
        "id": example["id"], "line": example["line"], "input_tokens": len(example["input_ids"]),
        "response_prediction_positions": len(positions), "transitions": transitions,
        "state_error_sq": state_error_sq, "teacher_state_sq": teacher_state_sq,
    }
    # Fail rather than silently reporting NaN/Inf or dropping a bad example.
    json.dumps(result, allow_nan=False)
    return result


def run_dataset(teacher, student, examples, teacher_layers, student_layers, args, path, label, torch, tqdm):
    rows = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), path.open("w") as stream:
        for example in tqdm(examples, desc=label, unit="example", dynamic_ncols=True):
            result = compare_example(teacher, student, example, teacher_layers, student_layers, args, torch)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            rows.append(result)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    tokens = sum(row["input_tokens"] for row in rows)
    return rows, {
        "evaluation_wall_seconds": seconds, "input_tokens": tokens,
        "input_tokens_per_second": tokens / seconds,
        "model_token_forwards": 2 * tokens,
        "model_token_forwards_per_second": 2 * tokens / seconds,
        "throughput_note": "Includes teacher + student forwards, projections, CPU transfer and JSONL IO; no generation.",
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def write_tables(output, summaries, comparisons, teacher_layers, student_layers):
    fields = ["dataset", "method", "transition", "examples", "transition_cosine_distance",
              "relative_transition_error", "relative_state_error", "degenerate_fraction"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for dataset, methods in summaries.items():
            for method, summary in methods.items():
                writer.writerow({"dataset": dataset, "method": method, "transition": "all", **summary["all"]})
                for j, stats in enumerate(summary["transitions"]):
                    writer.writerow({"dataset": dataset, "method": method, "transition": j + 1, **stats})
    lines = [
        "# Layer-change checkpoint diagnostic", "",
        "Space: shared-vocabulary log probabilities, not raw hidden states or continuous ODE velocities.",
        "Cosine: mean of per-example means over nondegenerate response transitions. Relative error: pooled squared-error / teacher squared-norm.",
        "Lower is better. Checkpoints are fixed; bootstrap CIs do not measure training-seed uncertainty.", "",
        f"Student hidden-state indices: `{student_layers}`; teacher: `{teacher_layers}`.", "",
        "These are transitions between selected layer pairs, not necessarily adjacent Transformer blocks.", "",
        "| Dataset | Method | Cosine distance ↓ | Relative transition error ↓ | Relative state error ↓ | Degenerate fraction |",
        "|---|---|---:|---:|---:|---:|",
    ]
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"
    for dataset, methods in summaries.items():
        for method, summary in methods.items():
            stats = summary["all"]
            lines.append(f"| {dataset} | {method} | " + " | ".join(fmt(stats[key]) for key in [
                "transition_cosine_distance", "relative_transition_error", "relative_state_error", "degenerate_fraction",
            ]) + " |")
    lines += ["", "## Paired differences", "", "Difference = augmented − baseline; a negative value favors augmentation.", ""]
    for dataset, pairs in comparisons.items():
        for pair, results in pairs.items():
            lines.append(f"### {dataset}: {pair}")
            for metric, result in results["all"].items():
                interval = result["ci95"]
                ci = f"[{fmt(interval[0])}, {fmt(interval[1])}]" if interval else "N/A"
                lines.append(f"- {metric}: {fmt(result['augmented_minus_baseline'])}; paired example-bootstrap 95% CI {ci}")
            lines.append("")
    lines += [
        "## Interpretation limits", "",
        "- FDD+CONTRA vs FDD tests incremental benefit, not standalone CONTRA vs FDD.",
        "- DistiLLM is not plain layer-wise state matching. No plain-state baseline is implied.",
        "- These metrics cannot establish the theorem's derivative-convergence premise.",
        "- Report null/negative and per-transition results as well as aggregates.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def make_plot(output, summaries):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"available": False, "reason": "matplotlib not installed; per-transition CSV/JSON retained"}
    fig, axes = plt.subplots(len(summaries), 2, figsize=(12, 4 * len(summaries)), squeeze=False)
    keys = ["transition_cosine_distance", "relative_transition_error"]
    for row, (dataset, methods) in enumerate(summaries.items()):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            for method, summary in methods.items():
                values = [s[key] if s[key] is not None else float("nan") for s in summary["transitions"]]
                ax.plot(range(1, len(values) + 1), values, marker="o", label=method)
            ax.set(title=f"{dataset}: {key}", xlabel="Aligned transition index", ylabel="Discrepancy (lower is better)")
            ax.legend()
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "transitions.png", dpi=160)
    plt.close(fig)
    return {"available": True, "path": "transitions.png"}


def main():
    args = parse_args()
    students, datasets = named_paths(args.student), named_paths(args.data)
    pairs = []
    for spec in args.compare:
        baseline, sep, augmented = spec.partition("=")
        if not sep or baseline not in students or augmented not in students or baseline == augmented:
            raise ValueError(f"Invalid comparison {spec!r}; expected two distinct --student names")
        pairs.append((baseline, augmented))
    teacher_path = Path(args.teacher).expanduser().resolve()
    teacher_info = model_metadata(teacher_path)
    student_info = {name: model_metadata(path) for name, path in students.items()}
    for path in datasets.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation data {path}. Provision download.txt first.")
    if args.num_distill_layers != 6:
        print("NOTE: non-default layer count; report as a separate diagnostic, not the training schedule.", flush=True)
    output = Path(args.output_root).expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {
        "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "teacher": teacher_info, "students": student_info,
        "python": platform.python_version(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "implementation_sha256": {p.name: file_hash(p) for p in [Path(__file__), Path(__file__).with_name("layer_change_metrics.py")]},
        "training": {"applicable": False, "reason": "Checkpoint-only evaluation; no training or optimizer"},
        "scope": "FDD-compatible vocabulary-space transitions; no plain-state baseline unless explicitly supplied",
        "telemetry": [],
    }
    write_json(output / "manifest.json", manifest)
    sampler, sampler_stream, sampler_info = None, None, None
    try:
        # Set before importing HF libraries. Never download on the evaluation host.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from tqdm import tqdm

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required. Launch with CUDA_VISIBLE_DEVICES=6.")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU via CUDA_VISIBLE_DEVICES")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest.update(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                        cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        dtype = getattr(torch, args.dtype)
        tokenizer = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError("Teacher tokenizer has no EOS token")
        teacher_layers, student_layers = None, None
        for name, info in student_info.items():
            t, s = uniform_schedule(
                teacher_info["config"]["n_layer"], info["config"]["n_layer"], args.num_distill_layers,
                allow_duplicates=args.allow_duplicate_layer_schedule,
            )
            if teacher_layers is not None and (t != teacher_layers or s != student_layers):
                raise ValueError("Use one student architecture per invocation for matched layer comparisons")
            teacher_layers, student_layers = t, s
            if info["config"]["vocab_size"] != teacher_info["config"]["vocab_size"]:
                raise ValueError(f"Vocabulary dimension mismatch: {name}")
            other_tokenizer = AutoTokenizer.from_pretrained(students[name], local_files_only=True)
            if other_tokenizer.get_vocab() != tokenizer.get_vocab() or other_tokenizer.eos_token_id != tokenizer.eos_token_id:
                raise ValueError(f"Vocabulary/token IDs differ for {name}; this is not a cross-tokenizer evaluator")
        manifest["schedule"] = {"teacher": teacher_layers, "student": student_layers}
        encoded, manifest["datasets"] = {}, {}
        for name, path in datasets.items():
            encoded[name], manifest["datasets"][name] = read_examples(path, tokenizer, args)
        sampler, sampler_stream, sampler_info = start_gpu_sampler(output)
        manifest["gpu_sampling"] = sampler_info
        write_json(output / "manifest.json", manifest)
        print(f"Output: {output}\nGPU: {manifest['gpu']}\nLayers: student={student_layers}, teacher={teacher_layers}", flush=True)
        load_start = time.perf_counter()
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path, torch_dtype=dtype, local_files_only=True, attn_implementation="eager",
        ).eval().to("cuda:0")
        torch.cuda.synchronize()
        manifest["teacher_load_seconds"] = time.perf_counter() - load_start
        summaries = {name: {} for name in datasets}
        all_records = {name: {} for name in datasets}
        for method, checkpoint in tqdm(list(students.items()), desc="Checkpoints", unit="model"):
            load_start = time.perf_counter()
            student = AutoModelForCausalLM.from_pretrained(
                checkpoint, torch_dtype=dtype, local_files_only=True, attn_implementation="eager",
            ).eval().to("cuda:0")
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - load_start
            for name, examples in encoded.items():
                rows, telemetry = run_dataset(
                    teacher, student, examples, teacher_layers, student_layers, args,
                    output / f"{name}__{method}.jsonl", f"{name}/{method}", torch, tqdm,
                )
                all_records[name][method] = rows
                summaries[name][method] = {
                    "all": aggregate(rows),
                    "transitions": [aggregate(rows, j) for j in range(len(teacher_layers) - 1)],
                }
                manifest["telemetry"].append({"dataset": name, "method": method,
                                              "student_load_seconds": load_seconds, **telemetry})
                write_json(output / "summary.json", summaries)
                write_json(output / "manifest.json", manifest)
            del student
            torch.cuda.empty_cache()
        comparisons = {name: {} for name in datasets}
        for name in tqdm(list(datasets), desc="Paired bootstrap", unit="dataset"):
            for baseline, augmented in pairs:
                a, b = all_records[name][baseline], all_records[name][augmented]
                comparisons[name][f"{baseline} -> {augmented}"] = {
                    "all": paired_bootstrap(a, b, args.bootstrap_samples, args.seed),
                    "transitions": [paired_bootstrap(a, b, args.bootstrap_samples, args.seed, j)
                                    for j in range(len(teacher_layers) - 1)],
                }
        write_json(output / "comparisons.json", comparisons)
        write_tables(output, summaries, comparisons, teacher_layers, student_layers)
        manifest["plot"] = make_plot(output, summaries)
        manifest["status"] = "completed"
        print((output / "report.md").read_text(), flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if sampler_info is not None:
            stop_gpu_sampler(sampler, sampler_stream, sampler_info)
        manifest["total_wall_seconds"] = time.perf_counter() - started
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
