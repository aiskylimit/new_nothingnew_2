"""Training logs of a positional-bias run (sgl.training.train --run-log-dir).

  config.json        hyperparameters, input hashes, environment
  data_check.json    token weights as the collator hands them to the loss; aborts on any mismatch
  train_log.jsonl    one line per optimizer step
  train_summary.json
"""

import importlib.metadata
import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from sgl.positional import common as C
from sgl.training.losses import IGNORE_INDEX


def data_check(dataset, collator, tokenizer, directory: Path) -> dict:
    n_supervised, mismatched, worst, samples = 0, 0, 0.0, []
    for index in range(len(dataset)):
        record = dataset[index]
        if "step_spans_w" not in record or "loss_weights" not in record:
            raise ValueError("--run-log-dir needs a dataset from sgl.positional.tables")
        batch = collator([record])
        weights, labels = batch["loss_weights"][0].double(), batch["labels"][0]
        expected = torch.zeros_like(weights)
        for first, last, weight in record["step_spans_w"]:
            expected[first:last] = weight
        end_of_steps = record["step_spans_w"][-1][1]
        expected[end_of_steps:] = 1.0  # stop token
        supervised = labels != IGNORE_INDEX
        n_supervised += int(supervised.sum())
        mismatched += int(((weights - expected).abs() > 1e-6)[supervised].sum())
        n_tokens = sum(last - first for first, last, _ in record["step_spans_w"])
        step_weight = sum(float(weights[first:last].sum()) for first, last, _ in record["step_spans_w"])
        worst = max(worst, abs(step_weight / n_tokens - 1))
        if index < 3:
            first = record["step_spans_w"][0][0]
            positions = [first + i * max(1, n_tokens // 20) for i in range(20) if first + i * max(1, n_tokens // 20) < end_of_steps]
            samples.append({"id": record["id"], "pairs": [
                [tokenizer.decode([record["input_ids"][p]]), round(float(weights[p]), 6)] for p in positions]})
    report = {"n_traces": len(dataset), "n_supervised_tokens": n_supervised,
              "max_abs(sum_token_weights/L - 1)": worst, "n_tokens_weight_mismatch": mismatched,
              "samples": samples, "provenance": C.provenance()}
    C.write_json(directory / "data_check.json", report)
    # float32 token weights: the per-trace sum is exact to about 1e-7
    if mismatched or worst > 1e-5:
        raise SystemExit(f"data check failed: {mismatched} mismatched tokens, mass error {worst:.2e}")
    return report


def write_run_config(directory: Path, config: dict, training_args, model) -> None:
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    root = directory.parents[1]  # results/positional/runs/<id> -> results/positional
    name = Path(config["data_path"]).stem.rsplit("-", 1)[-1]
    weights = root / f"weights/w_{name}.parquet"
    deepspeed = config.get("deepspeed_config")
    C.write_json(directory / "config.json", {
        "hyperparameters": {
            "epochs": training_args.num_train_epochs,
            "micro_batch": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "effective_batch": training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps * training_args.world_size,
            "learning_rate": training_args.learning_rate, "min_learning_rate": config.get("min_learning_rate"),
            "scheduler": str(training_args.lr_scheduler_type), "warmup_ratio": config.get("warmup_ratio"),
            "max_length": config.get("max_seq_len"), "optimizer": str(training_args.optim),
            "zero_stage": json.loads(Path(deepspeed).read_text())["zero_optimization"]["stage"] if deepspeed else None,
            "dtype": str(next(model.parameters()).dtype), "bf16": training_args.bf16,
            "seed": training_args.seed, "attn_implementation": config.get("attn_implementation"),
        },
        "model": config["model_name"],
        "dataset_file": config["data_path"], "dataset_sha256": C.sha256(config["data_path"]),
        "weights_file": str(weights) if weights.is_file() else None,
        "weights_sha256": C.sha256(weights) if weights.is_file() else None,
        "environment": {
            **{p: version(p) for p in ("torch", "transformers", "deepspeed", "peft", "accelerate")},
            "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
        "provenance": C.provenance(),
    })


class StepLogger(TrainerCallback):
    """Accumulates the token sums of every micro-batch and writes one line per optimizer step."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "train_log.jsonl"
        self.path.write_text("")
        self.started = time.time()
        self.last = None
        self.n_steps = 0
        self._reset()

    def _reset(self):
        self.sums = {}
        self.pending = None
        self.step_start = time.time()

    def add(self, logits, labels, weights, source, denominator) -> None:
        from sgl.training.baselines import per_token_ce

        with torch.no_grad():
            ce = per_token_ce(logits.detach(), labels).double()
            target = labels[:, 1:]
            valid = target != IGNORE_INDEX
            w = weights[:, 1:].double()
            values = {"weighted": (w * ce)[valid].sum(), "ce": ce[valid].sum(), "w": w[valid].sum(),
                      "n": valid.sum(), "input_tokens": torch.tensor(labels.numel())}
            if source is not None:
                for name, code in (("prefix", 1), ("cont", 2)):
                    part = valid & (source[:, 1:] == code)
                    values[f"ce_{name}"], values[f"n_{name}"] = ce[part].sum(), part.sum()
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
        if denominator is not None:
            self.sums["Z"] = float(denominator)

    def on_step_begin(self, args, state, control, **kwargs):
        self._reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        s = self.sums
        Z = s.get("Z") or s["n"]
        ratio = lambda a, b: a / b if b else None  # noqa: E731
        self.pending = {
            "global_step": state.global_step, "epoch": state.epoch, "loss_weighted": s["weighted"] / Z,
            "nll_unweighted": ratio(s["ce"], s["n"]), "nll_prefix": ratio(s.get("ce_prefix", 0), s.get("n_prefix", 0)),
            "nll_cont": ratio(s.get("ce_cont", 0), s.get("n_cont", 0)), "Z": Z, "sum_w_over_Z": s["w"] / Z,
            "tokens_per_sec": s["input_tokens"] / max(time.time() - self.step_start, 1e-9),
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pending is None or not logs or "loss" not in logs:
            return
        # grad_norm is the pre-clip norm the Trainer (or DeepSpeed) reports
        line = {**self.pending, "lr": logs.get("learning_rate"), "grad_norm": logs.get("grad_norm")}
        with self.path.open("a") as handle:
            handle.write(json.dumps(line) + "\n")
        self.last = line
        self.n_steps += 1
        self.pending = None

    def write_summary(self, checkpoint: str, gpus: int) -> None:
        hours = (time.time() - self.started) / 3600
        lines = [json.loads(x) for x in self.path.read_text().splitlines()]
        final_epoch = [x for x in lines if x["epoch"] > lines[-1]["epoch"] - 1] if lines else []
        mean = lambda key: sum(x[key] for x in final_epoch) / len(final_epoch) if final_epoch else None  # noqa: E731
        C.write_json(self.directory / "train_summary.json", {
            "final_nll_unweighted": mean("nll_unweighted"), "final_loss_weighted": mean("loss_weighted"),
            "final_means_over": "optimizer steps of the last epoch", "wall_clock_hours": hours,
            "gpu_hours": hours * gpus, "n_optimizer_steps": self.n_steps, "checkpoint": checkpoint,
            "provenance": C.provenance(),
        })
