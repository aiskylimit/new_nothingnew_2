"""Training logs of a positional-bias run (sgl.training.train --run-log-dir).

  config.json        hyperparameters, input hashes, environment
  data_check.json    token weights as the collator hands them to the loss; aborts on any mismatch
  train_log.jsonl    one line per optimizer step
  train_summary.json
"""

import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from sgl.positional import common as C
from sgl.training.losses import IGNORE_INDEX

# token_tags bits of sgl.robustness.signals: 1 prefix, 2 continuation, 4 junction, 8 answer step, 16 stop
# token, 32 any step token
TAG_REGIONS = ("junction", "answer_steps", "non_answer_steps", "eos")


def region_masks(tags: torch.Tensor, valid: torch.Tensor) -> dict:
    step = (tags & 32) > 0
    answer = (tags & 8) > 0
    return {"junction": valid & ((tags & 4) > 0), "answer_steps": valid & step & answer,
            "non_answer_steps": valid & step & ~answer, "eos": valid & ((tags & 16) > 0)}


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
        for name in ("health.jsonl", "epoch_log.jsonl"):  # a rerun starts these afresh too
            (self.directory / name).unlink(missing_ok=True)
        self.started = time.time()
        self.last = None
        self.n_steps = 0
        self.grad_norms = []
        self._reset()

    def _reset(self):
        self.sums = {}
        self.pending = None
        self.step_start = time.time()

    def add(self, logits, labels, weights, source, denominator, tags=None, input_ids=None) -> None:
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
            if tags is not None:
                for name, part in region_masks(tags[:, 1:], valid).items():
                    values[f"ce_{name}"], values[f"n_{name}"] = ce[part].sum(), part.sum()
            w_max = float(w[valid].max()) if bool(valid.any()) else 0.0
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.sums["w_max"] = max(self.sums.get("w_max", 0.0), w_max)
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
            **{f"nll_{name}": ratio(s.get(f"ce_{name}", 0), s.get(f"n_{name}", 0))
               for name in TAG_REGIONS if f"n_{name}" in s},
            "w_mean": ratio(s["w"], s["n"]), "w_max": s.get("w_max"),
            "tokens_per_sec": s["input_tokens"] / max(time.time() - self.step_start, 1e-9),
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pending is None or not logs or "loss" not in logs:
            return
        # grad_norm is the pre-clip norm the Trainer (or DeepSpeed) reports
        line = {**self.pending, "lr": logs.get("learning_rate"), "grad_norm": logs.get("grad_norm")}
        line["loss_nll_unweighted"] = line["nll_unweighted"]
        self._health(line)
        with self.path.open("a") as handle:
            handle.write(json.dumps(line) + "\n")
        self.last = line
        self.n_steps += 1
        self.pending = None

    def _health(self, line: dict) -> None:
        """Warn (health.jsonl + stdout) on a non-finite loss / grad norm or a grad-norm spike."""
        problems = []
        for key in ("loss_weighted", "nll_unweighted", "grad_norm"):
            value = line.get(key)
            if value is not None and not math.isfinite(value):
                problems.append(f"{key} is {value}")
        norm = line.get("grad_norm")
        history = self.grad_norms[-50:]
        if norm is not None and math.isfinite(norm):
            if len(history) >= 10 and norm > 10 * sorted(history)[len(history) // 2]:
                problems.append(f"grad_norm {norm:.3g} > 10x the median of the last {len(history)} steps")
            self.grad_norms.append(norm)
        if abs(line.get("sum_w_over_Z", 1.0) - 1.0) > 0.05 and line.get("Z"):
            problems.append(f"sum(w)/Z = {line['sum_w_over_Z']:.3f}")
        for problem in problems:
            print(f"WARNING step {line['global_step']}: {problem}")
            with (self.directory / "health.jsonl").open("a") as handle:
                handle.write(json.dumps({"global_step": line["global_step"], "warning": problem}) + "\n")

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


class ParamChangeCheck(TrainerCallback):
    """Share of parameters that moved away from the base model at optimizer step `step` (Appendix A:
    bf16 without fp32 master weights changed only 13.3% of them and the loss did not fall)."""

    def __init__(self, directory: Path, step: int = 64):
        self.directory, self.step, self.initial = Path(directory), step, None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.initial = {
            name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters() if p.requires_grad
        }

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step != self.step or self.initial is None:
            return
        changed = total = 0
        for name, p in model.named_parameters():
            if name in self.initial:
                changed += int((p.detach().cpu() != self.initial[name]).sum())
                total += p.numel()
        share = changed / max(total, 1)
        report = {"global_step": self.step, "changed_fraction": share, "n_params": total}
        if share < 0.5:
            report["warning"] = "less than half of the parameters changed: check the fp32 master weights"
            print(f"WARNING: only {share:.1%} of the parameters changed by step {self.step}")
        write_json_line(self.directory / "health.jsonl", report)
        self.initial = None  # free the host copy


class EpochEnd(TrainerCallback):
    """At the end of every epoch: unweighted NLL of the current model on the whole training set, split by
    token_tags region (epoch_log.jsonl, with the means of that epoch's step lines), and a bf16
    weights-only checkpoint in <checkpoint_dir>/epoch<e> for every epoch but the last (the last one is
    the run's final checkpoint)."""

    def __init__(self, directory: Path, dataset, tokenizer, step_logger: "StepLogger | None",
                 checkpoint_dir: str | None, nll: bool, epochs: int):
        self.directory, self.dataset, self.tokenizer = Path(directory), dataset, tokenizer
        self.step_logger, self.checkpoint_dir, self.nll, self.epochs = step_logger, checkpoint_dir, nll, epochs

    @torch.no_grad()
    def dataset_nll(self, model) -> dict:
        from sgl.signals.predictability import token_log_probabilities

        was_training = model.training
        model.eval()
        sums, device = {}, next(model.parameters()).device
        inner = model.get_base_model() if hasattr(model, "get_base_model") else model
        unembedding = inner.get_output_embeddings().weight
        for index in range(len(self.dataset)):
            record = self.dataset[index]
            ids = torch.tensor([record["input_ids"]], device=device)
            mask = torch.tensor(record["loss_mask"][1:], device=device).bool()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = inner.model(input_ids=ids).last_hidden_state[0, :-1]
            nll = -token_log_probabilities(hidden, ids[0, 1:], unembedding).double()
            parts = {"all": mask}
            if "token_tags" in record:
                tags = torch.tensor(record["token_tags"][1:], device=device)
                parts |= {"prefix": mask & ((tags & 1) > 0), "continuation": mask & ((tags & 2) > 0)}
                parts |= region_masks(tags, mask)
            for name, part in parts.items():
                sums[f"ce_{name}"] = sums.get(f"ce_{name}", 0.0) + float(nll[part].sum())
                sums[f"n_{name}"] = sums.get(f"n_{name}", 0) + int(part.sum())
        if was_training:
            model.train()
        return {f"dataset_nll_{name[3:]}": sums[name] / sums[f"n_{name[3:]}"]
                for name in sums if name.startswith("ce_") and sums[f"n_{name[3:]}"]}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(round(state.epoch))
        line = {"epoch": epoch, "global_step": state.global_step}
        if self.step_logger is not None and self.step_logger.path.is_file():
            steps = [json.loads(x) for x in self.step_logger.path.read_text().splitlines()]
            steps = [x for x in steps if epoch - 1 < x["epoch"] <= epoch + 1e-6]
            for key in {k for x in steps for k, v in x.items() if isinstance(v, (int, float)) and v is not None}:
                values = [x[key] for x in steps if isinstance(x.get(key), (int, float))]
                if values and key not in ("epoch", "global_step"):
                    line[f"mean_{key}"] = sum(values) / len(values)
        started = time.time()
        if self.nll:
            line |= self.dataset_nll(model)
            line["dataset_nll_seconds"] = time.time() - started
        if self.checkpoint_dir and epoch < self.epochs and state.is_world_process_zero:
            target = Path(self.checkpoint_dir) / f"epoch{epoch}"
            inner = model.get_base_model() if hasattr(model, "get_base_model") else model
            inner.save_pretrained(target, state_dict={k: v.to(torch.bfloat16) for k, v in inner.state_dict().items()})
            self.tokenizer.save_pretrained(target)
            line["checkpoint"] = str(target)
        if state.is_world_process_zero:
            write_json_line(self.directory / "epoch_log.jsonl", line)
            nll = ", ".join(f"{k}={v:.4f}" for k, v in line.items() if k.startswith("dataset_nll_"))
            print(f"epoch {epoch}: {nll}")


def write_json_line(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(data) + "\n")
