"""Training logs of an ALG-hard run (sgl.training.train --run-log-dir D --selection-log).

On top of sgl.positional.runlog: the NLL of kept and dropped tokens from the same forward (detached), token
counts, grad clipping, per-epoch NLL over every response token (also before training, epoch 0) and a data
check of the token weights the collator hands to the loss.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sgl.hardsel import common as C
from sgl.positional.runlog import EpochEnd, StepLogger, write_json_line
from sgl.training.losses import IGNORE_INDEX

# region -> token_tags bit
REGIONS = {"prefix": 1, "cont": 2, "junction": 4, "answer_steps": 8, "eos": 16, "kept": C.TAG_KEPT,
           "dropped": C.TAG_DROPPED, "h59_dropped": C.TAG_H59_DROPPED}
RESPONSE_BITS = 16 | 32  # stop token | step token


def region_masks(tags: torch.Tensor) -> dict[str, torch.Tensor]:
    response = (tags & RESPONSE_BITS) > 0
    out = {"all": response} | {name: response & ((tags & bit) > 0) for name, bit in REGIONS.items()}
    out["non_answer_steps"] = ((tags & 32) > 0) & ((tags & 8) == 0)
    return out


def data_check(dataset, collator, directory: Path) -> dict:
    """Token weights and mask as the loss receives them must equal the training file."""
    loaded, stored = hashlib.sha256(), hashlib.sha256()
    mismatched, worst_mass, n_supervised, first50 = 0, 0.0, 0, 0
    for index in range(len(dataset)):
        record = dataset[index]
        batch = collator([record])
        weights, labels = batch["loss_weights"][0].numpy(), batch["labels"][0].numpy()
        mask = (labels != IGNORE_INDEX).astype(np.uint8)
        expected_w = np.asarray(record["loss_weights"], dtype=np.float32)
        expected_m = np.asarray(record["loss_mask"], dtype=np.uint8)
        different = int((weights != expected_w).sum() + (mask != expected_m).sum())
        mismatched += different
        first50 += different if index < 50 else 0
        loaded.update(weights.tobytes() + mask.tobytes())
        stored.update(expected_w.tobytes() + expected_m.tobytes())
        n_supervised += int(mask.sum())
        # weights live only on supervised tokens and keep their count (binary, or ALG over the kept steps)
        worst_mass = max(worst_mass, abs(float(expected_w.astype(np.float64).sum()) / int(mask.sum()) - 1),
                         float(np.abs(expected_w[expected_m == 0]).max(initial=0.0)))
    report = {"n_traces": len(dataset), "n_supervised_tokens": n_supervised, "n_mismatched_tokens": mismatched,
              "n_mismatched_tokens_first_50_traces": first50, "loaded_sha256": loaded.hexdigest(),
              "file_sha256_of_weights_and_mask": stored.hexdigest(), "max_abs(sum_w/n_supervised - 1)": worst_mass,
              "provenance": C.provenance()}
    C.write_json(Path(directory) / "data_check.json", report)
    if mismatched or loaded.hexdigest() != stored.hexdigest() or worst_mass > 1e-5:
        raise SystemExit(f"data check failed: {mismatched} mismatched tokens, mass error {worst_mass:.2e}")
    return report


def expected_kept_frac(dataset) -> float:
    kept = total = 0
    for index in range(len(dataset)):
        tags = np.asarray(dataset[index]["token_tags"])
        total += int(((tags & 32) > 0).sum())
        kept += int(((tags & C.TAG_KEPT) > 0).sum())
    return kept / total


class SelectionStepLogger(StepLogger):
    def __init__(self, directory: Path, dataset, max_grad_norm: float = 1.0):
        self.expected = expected_kept_frac(dataset)
        self.max_grad_norm = max_grad_norm
        super().__init__(directory)

    def add(self, logits, labels, weights, source, denominator, tags=None, input_ids=None) -> None:
        from sgl.training.baselines import per_token_ce

        super().add(logits, labels, weights, None, denominator)
        with torch.no_grad():
            masks = region_masks(tags)
            everything = torch.where(masks["all"], input_ids, torch.full_like(input_ids, IGNORE_INDEX))
            ce = per_token_ce(logits.detach(), everything).double()
            for name, part in masks.items():
                part = part[:, 1:]
                self.sums[f"h_ce_{name}"] = self.sums.get(f"h_ce_{name}", 0.0) + float(ce[part].sum())
                self.sums[f"h_n_{name}"] = self.sums.get(f"h_n_{name}", 0.0) + float(part.sum())

    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)
        s = self.sums
        nll = lambda name: s[f"h_ce_{name}"] / s[f"h_n_{name}"] if s.get(f"h_n_{name}") else None  # noqa: E731
        n_kept, n_dropped = s["h_n_kept"], s["h_n_dropped"]
        self.pending |= {
            "step": state.global_step,
            "loss": self.pending["loss_weighted"],
            "nll_kept": self.pending["nll_unweighted"],  # every supervised token: kept steps + stop token
            "nll_dropped": nll("dropped"),
            "nll_all_unweighted": nll("all"),
            **{f"nll_{name}": nll(name) for name in
               ("prefix", "cont", "junction", "answer_steps", "non_answer_steps", "eos", "h59_dropped")},
            "n_tok_total": int(n_kept + n_dropped),  # step tokens, stop token excluded
            "n_tok_kept": int(n_kept),
            "kept_frac_batch": n_kept / (n_kept + n_dropped) if n_kept + n_dropped else None,
            "step_time_s": time.time() - self.step_start,
        }

    def _health(self, line: dict) -> None:
        norm = line.get("grad_norm")
        line["grad_clipped"] = bool(norm is not None and norm > self.max_grad_norm)
        super()._health(line)
        problems = []
        expected_tokens = self.expected * line["n_tok_total"]
        if line["n_tok_kept"] < 0.1 * expected_tokens:
            problems.append(f"only {line['n_tok_kept']} kept tokens, expected about {expected_tokens:.0f}")
        if line["kept_frac_batch"] is not None and abs(line["kept_frac_batch"] - self.expected) > 0.05:
            problems.append(f"kept_frac_batch {line['kept_frac_batch']:.3f} vs corpus {self.expected:.3f}")
        if abs(line["sum_w_over_Z"] - 1) > 1e-3:
            problems.append(f"sum_w_over_Z = {line['sum_w_over_Z']:.5f}")
        for problem in problems:
            print(f"WARNING step {line['global_step']}: {problem}")
            write_json_line(self.directory / "health.jsonl", {"global_step": line["global_step"], "warning": problem})

    def write_summary(self, checkpoint: str, gpus: int) -> None:
        super().write_summary(checkpoint, gpus)
        lines = [json.loads(x) for x in self.path.read_text().splitlines()]
        summary = C.read_json(self.directory / "train_summary.json")
        last = [x for x in lines if x["epoch"] > lines[-1]["epoch"] - 1] if lines else []
        for key in ("loss", "nll_kept", "nll_dropped", "nll_all_unweighted"):
            values = [x[key] for x in last if x.get(key) is not None]
            summary[f"final_{key}"] = sum(values) / len(values) if values else None
        peaks = [x["max_mem_gb"] for x in lines if x.get("max_mem_gb") is not None]
        summary |= {"peak_mem_gb": max(peaks) if peaks else None, "expected_kept_frac": self.expected}
        C.write_json(self.directory / "train_summary.json", summary)


@torch.no_grad()
def region_nll(model, dataset) -> dict:
    """Unweighted NLL of the model on every response token of the dataset, by region."""
    from sgl.signals.predictability import token_log_probabilities

    was_training = model.training
    model.eval()
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    inner = inner.get_base_model() if hasattr(inner, "get_base_model") else inner
    device = next(inner.parameters()).device
    unembedding = inner.get_output_embeddings().weight
    sums, counts = {}, {}
    for index in range(len(dataset)):
        record = dataset[index]
        ids = torch.tensor([record["input_ids"]], device=device)
        tags = torch.tensor(record["token_tags"][1:], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = inner.model(input_ids=ids).last_hidden_state[0, :-1]
        nll = -token_log_probabilities(hidden, ids[0, 1:], unembedding).double()
        for name, part in region_masks(tags).items():
            sums[name] = sums.get(name, 0.0) + float(nll[part].sum())
            counts[name] = counts.get(name, 0) + int(part.sum())
    if was_training:
        model.train()
    return {f"dataset_nll_{name}": sums[name] / counts[name] for name in sums if counts[name]}


class SelectionEpochEnd(EpochEnd):
    def dataset_nll(self, model) -> dict:
        return region_nll(model, self.dataset)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self.nll:
            return
        started = time.time()
        line = {"epoch": 0, "global_step": 0, **self.dataset_nll(model)}
        line["dataset_nll_seconds"] = time.time() - started
        if state.is_world_process_zero:
            write_json_line(self.directory / "epoch_log.jsonl", line)
            print(f"epoch 0 (before training): nll all {line['dataset_nll_all']:.4f}")
