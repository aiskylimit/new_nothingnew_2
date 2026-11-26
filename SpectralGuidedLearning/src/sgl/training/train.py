"""Phase 5: masked supervised fine-tuning (paper Eq. 9).

Shared by both the vanilla and spectral SFT launchers (scripts/qwen3/{sft,spectral}/) --
only the loss mask in the dataset differs. Settings come from CLI flags or --config.

--trans-lambda > 0 adds the next-step representation objective (L = L_NLL + lambda * L_trans,
see sgl.training.transition_loss) on top of whichever NLL mask the data carries; the data then needs the
step fields written by sgl.transforms.build_transitions. Off by default, so every existing arm is unchanged.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from sgl.hardsel import runlog as selection_log
from sgl.positional.runlog import EpochEnd, ParamChangeCheck, StepLogger, data_check, write_run_config
from sgl.training.baselines import AnchoredMixin, VCoreMixin
from sgl.training.collator import MaskedSFTCollator
from sgl.training.dataset import MaskedSFTDataset
from sgl.training.losses import OBJECTIVE_NLL, OBJECTIVES, masked_cross_entropy
from sgl.training.transition_loss import default_transition_layer, find_decoder_layers
from sgl.training.transition_trainer import TRANS_INPUT_KEYS, TransitionLossMixin, attach_predictor
from sgl.training.utils import compensate_global_token_mean, set_training_seed


class MaskedSFTTrainer(Trainer):
    """Trainer applying the selective masked objective instead of the default LM loss.

    `model_accepts_loss_kwargs=True` makes Trainer sum supervised-token counts across a whole
    gradient-accumulation step and pass it as `num_items_in_batch`, used as Z in Eq. 9 --
    averaging per-microbatch losses instead would over-weight sparsely-supervised sequences
    (exactly what spectral masks create).
    """

    def __init__(
        self, *args, objective="nll", prefix_objective=None, cont_objective=None, junction_objective=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Per-token objective: `objective` everywhere, optionally overridden on teacher-prefix
        # (token_source 1) / continuation (token_source 2) tokens -- provenance-aware SFT.
        self.objective = OBJECTIVES[objective]
        self.source_objectives = {
            source: OBJECTIVES[name]
            for source, name in ((1, prefix_objective), (2, cont_objective), (3, junction_objective))
            if name is not None
        }
        self.step_logger = None  # sgl.positional.runlog.StepLogger (--run-log-dir)
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True  # our loss shifts labels, so count over labels[..., 1:]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        token_weights = inputs.pop("loss_weights", None)
        token_source = inputs.pop("token_source", None)
        token_tags = inputs.pop("token_tags", None)  # logging only (sgl.robustness.signals)
        token_objective = None
        if self.source_objectives and token_source is None:
            raise ValueError(
                "--prefix/--cont/--junction-objective need token_source (sgl.transforms.provenance)"
            )
        if self.objective != OBJECTIVE_NLL or self.source_objectives:
            token_objective = torch.full_like(labels, self.objective)
            for source, objective in self.source_objectives.items():
                token_objective[token_source == source] = objective
        for key in TRANS_INPUT_KEYS:  # step structure of a *-trans.jsonl is inert without --trans-lambda
            inputs.pop(key, None)
        outputs = model(**inputs)
        if self.step_logger is not None:
            self.step_logger.add(
                outputs.logits, labels, token_weights, token_source, num_items_in_batch, token_tags,
                input_ids=inputs["input_ids"],
            )
        loss = masked_cross_entropy(
            outputs.logits, labels, denominator=num_items_in_batch, token_weights=token_weights,
            token_objective=token_objective,
        )
        # ``num_items_in_batch`` is a token count gathered across DDP ranks.  Gradients are
        # averaged by DDP, so mirror Trainer.compute_loss's compensation and recover the
        # global sum/Z objective instead of shrinking it by world size.
        loss = compensate_global_token_mean(
            loss,
            average_tokens_across_devices=self.args.average_tokens_across_devices,
            num_items_in_batch=num_items_in_batch,
            num_processes=self.accelerator.num_processes,
        )
        return (loss, outputs) if return_outputs else loss


class TransitionSFTTrainer(TransitionLossMixin, MaskedSFTTrainer):
    """MaskedSFTTrainer + lambda * L_trans (SFT + L_trans, or SGL + L_trans with a spectral mask)."""


class AnchoredSFTTrainer(AnchoredMixin, MaskedSFTTrainer):
    """ASFT: DFT plus a KL anchor to the pretrained student (sgl.training.baselines)."""


class VCoreSFTTrainer(VCoreMixin, MaskedSFTTrainer):
    """VCORE: per-token weights from a one-step probe (sgl.training.baselines)."""


def build_training_arguments(config: dict) -> TrainingArguments:
    on_gpu = torch.cuda.is_available()  # CPU falls back to fp32 for local smoke runs
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        # min_lr_rate (not min_lr) -- min_lr makes transformers compute the ratio via
        # optimizer.defaults["lr"], which DeepSpeed's wrapped ZeRO optimizer doesn't expose
        # (AttributeError: 'DeepSpeedZeroOptimizer' object has no attribute 'defaults').
        lr_scheduler_kwargs={"min_lr_rate": config["min_learning_rate"] / config["learning_rate"]},
        warmup_ratio=config["warmup_ratio"],  # deprecated in transformers>=5 (still functions correctly)
        # VCORE autocasts by itself: its scoring forwards must stay fp32
        bf16=on_gpu and config.get("baseline") != "vcore",
        use_cpu=not on_gpu,
        gradient_checkpointing=on_gpu,
        # Reentrant checkpointing trips DDP's "mark variable ready only once" assertion with
        # LoRA's frozen base params (multi-GPU only); non-reentrant avoids the double-backward.
        gradient_checkpointing_kwargs={"use_reentrant": False} if on_gpu else None,
        logging_steps=config.get("logging_steps", 5),
        # save_strategy: "epoch"/"steps" as usual; "no" skips intermediate checkpoints (final save still runs).
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 6),
        report_to=[],
        seed=config.get("seed", 42),
        remove_unused_columns=False,
        deepspeed=config.get("deepspeed_config"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # --config is an optional yaml base; CLI flags below override it when both are given.
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate the setup")
    parser.add_argument("--model-name")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--attn-implementation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument(
        "--metrics-log",
        help="write the full training metric history (trainer.state.log_history: per-step loss, "
        "grad_norm, learning_rate, epoch + final summary) to this JSON file after training",
    )
    parser.add_argument(
        "--deepspeed-config",
        help="path to a DeepSpeed json config (e.g. configs/deepspeed/ds_config_zero2_offload.json); "
        "batch size/grad accum/bf16 fields there are \"auto\", filled in from the flags above. "
        "ZeRO-2 + optimizer CPU offload for long-sequence runs that OOM at --per-device-batch-size 1.",
    )
    parser.add_argument(
        "--max-seq-len", type=int,
        help="drop samples with more than this many input_ids before training. Optimizer-state "
        "offload (--deepspeed-config) doesn't reduce per-sample activation memory -- a single very "
        "long sequence at --per-device-batch-size 1 can still OOM regardless. Empirically measured "
        "on a 40GB GPU (Qwen3-1.7B, this LoRA config, gradient checkpointing on): ~31GB peak at "
        "12k tokens, ~40GB (no margin) at 16k, OOM at 20k -- pick a value with headroom for your GPU.",
    )
    # checkpointing
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"])
    parser.add_argument("--save-steps", type=int, help="interval when --save-strategy=steps")
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction,
        help="resume from the last checkpoint in --output-dir if one exists (else start fresh)",
    )
    # LoRA (optional): trains low-rank adapters instead of full weights. Merged back into the
    # base model before saving, so output_dir is a normal full checkpoint either way.
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--model-dtype", choices=("bfloat16", "float32"),
        help="dtype the weights are loaded in (default bfloat16, as every earlier run). float32 keeps "
        "fp32 weights and relies on Trainer bf16 autocast for compute (true mixed precision); use it "
        "without --deepspeed-config, since DeepSpeed's bf16 engine would cast the module back to bf16",
    )
    # Per-token objective (default nll = plain masked SFT). dft = -sg(p) log p (arXiv 2508.05629).
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), help="objective on every supervised token")
    parser.add_argument(
        "--prefix-objective", choices=sorted(OBJECTIVES),
        help="override on teacher-prefix tokens (token_source 1, sgl.transforms.provenance)",
    )
    parser.add_argument(
        "--cont-objective", choices=sorted(OBJECTIVES),
        help="override on continuation tokens (token_source 2, sgl.transforms.provenance)",
    )
    parser.add_argument(
        "--junction-objective", choices=sorted(OBJECTIVES),
        help="override on junction tokens (token_source 3: first continuation tokens after the prefix)",
    )
    parser.add_argument(
        "--lora-merge", action=argparse.BooleanOptionalAction,
        help="merge adapters into the base weights before saving (default true); "
        "--no-lora-merge keeps output_dir as an adapter-only checkpoint (much smaller, "
        "loadable by vLLM's native LoRA support without a second full-weight copy)",
    )
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules", help="comma-separated module names")
    # Published token-reweighting baselines (sgl.training.baselines); unset = masked SFT as above.
    parser.add_argument("--baseline", choices=("asft", "vcore"))
    parser.add_argument("--asft-kl-lambda", type=float, help="ASFT KL anchor weight (default 0.05)")
    parser.add_argument("--vcore-epsilon", type=float, help="VCORE probe step size (default 1e-4)")
    parser.add_argument("--vcore-temperature", type=float, help="VCORE temperature (default 5000)")
    parser.add_argument(
        "--vcore-probe-batches", type=int,
        help="samples in VCORE's random batch B' (default: the gradient accumulation steps)",
    )
    parser.add_argument(
        "--run-log-dir",
        help="write config.json, data_check.json, a per-optimizer-step train_log.jsonl and "
        "train_summary.json there (sgl.positional.runlog; forces logging every step)",
    )
    parser.add_argument(
        "--epoch-nll", action="store_true",
        help="with --run-log-dir: unweighted NLL of the whole training set at the end of every epoch, "
        "split by token_tags region (epoch_log.jsonl)",
    )
    parser.add_argument(
        "--epoch-checkpoints-dir",
        help="with --run-log-dir: bf16 weights-only checkpoint <dir>/epoch<e> at the end of every epoch but the last",
    )
    parser.add_argument(
        "--param-check-step", type=int,
        help="with --run-log-dir: share of parameters changed from the base model at this optimizer step",
    )
    parser.add_argument(
        "--selection-log", action="store_true",
        help="with --run-log-dir: the ALG-hard logs (sgl.hardsel.runlog): NLL of kept / dropped tokens, "
        "per-epoch NLL over every response token, also before training",
    )
    # Next-step representation prediction (L_trans). Only --trans-lambda switches it on.
    parser.add_argument(
        "--trans-lambda", type=float,
        help="weight lambda of L_trans (0/unset = plain masked SFT). Needs a dataset with the step "
        "fields from sgl.transforms.build_transitions",
    )
    parser.add_argument(
        "--trans-layer", type=int,
        help="0-based decoder layer whose output feeds source and target (default: round(2/3 * depth), "
        "24 for Qwen3-8B's 36 layers)",
    )
    parser.add_argument("--trans-hidden", type=int, help="predictor bottleneck width (default 1024)")
    parser.add_argument(
        "--trans-lr", type=float,
        help="learning rate of the predictor's own param group, no weight decay (default 2e-4)",
    )
    parser.add_argument(
        "--trans-lambda-warmup", action=argparse.BooleanOptionalAction,
        help="ramp lambda linearly from 0 over the LR warmup steps (default on)",
    )
    parser.add_argument(
        "--trans-shuffle-targets", action=argparse.BooleanOptionalAction,
        help="control arm: target is a random other step of the same trace instead of step i+1",
    )
    parser.add_argument(
        "--trans-grad-log-interval", type=int,
        help="on every Nth optimizer step log ||dL_NLL/dH|| vs lambda*||dL_trans/dH|| at the "
        "transition layer (tensor hooks in the real backward, no extra passes; default 50, 0 = off)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "logging_steps": args.logging_steps,
        "metrics_log": args.metrics_log,
        "deepspeed_config": args.deepspeed_config,
        "max_seq_len": args.max_seq_len,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "resume": args.resume,
        "use_lora": args.use_lora,
        "model_dtype": args.model_dtype,
        "objective": args.objective,
        "prefix_objective": args.prefix_objective,
        "cont_objective": args.cont_objective,
        "junction_objective": args.junction_objective,
        "lora_merge": args.lora_merge,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "baseline": args.baseline,
        "asft_kl_lambda": args.asft_kl_lambda,
        "vcore_epsilon": args.vcore_epsilon,
        "vcore_temperature": args.vcore_temperature,
        "vcore_probe_batches": args.vcore_probe_batches,
        "run_log_dir": args.run_log_dir,
        "epoch_nll": args.epoch_nll or None,
        "epoch_checkpoints_dir": args.epoch_checkpoints_dir,
        "param_check_step": args.param_check_step,
        "selection_log": args.selection_log or None,
        "trans_lambda": args.trans_lambda,
        "trans_layer": args.trans_layer,
        "trans_hidden": args.trans_hidden,
        "trans_lr": args.trans_lr,
        "trans_lambda_warmup": args.trans_lambda_warmup,
        "trans_shuffle_targets": args.trans_shuffle_targets,
        "trans_grad_log_interval": args.trans_grad_log_interval,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    use_trans = bool(config.get("trans_lambda"))

    baseline = config.get("baseline")
    if baseline:
        clashes = [
            key for key in ("trans_lambda", "prefix_objective", "cont_objective", "junction_objective")
            if config.get(key)
        ]
        if config.get("objective") not in (None, "nll"):
            clashes.append("objective")
        if baseline == "asft" and config.get("use_lora"):
            clashes.append("use_lora")  # the reference would need adapters disabled instead of a copy
        if clashes:
            parser.error(f"--baseline {baseline} defines its own loss and cannot be combined with {clashes}")

    missing = [key for key in ("model_name", "data_path", "output_dir") if key not in config]
    if missing:
        parser.error(f"missing required settings (pass via --config or CLI flags): {missing}")

    # Trainer sets this later in its constructor, which is too late for LoRA's random
    # adapter initialization.  Do it before loading or wrapping the model.
    set_training_seed(config.get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = MaskedSFTDataset(config["data_path"], max_seq_len=config.get("max_seq_len"))

    if args.smoke:
        dataset.records = dataset.records[:8]
        config["epochs"] = 1

    print(f"{len(dataset)} samples, {dataset.supervised_token_count():,} supervised tokens")
    if use_trans:
        missing_fields = [r.get("id") for r in dataset.records if "step_id" not in r][:5]
        if missing_fields:
            parser.error(
                "--trans-lambda needs step fields (sgl.transforms.build_transitions); "
                f"records without them: {missing_fields}..."
            )
        pairs = sum(len(r["pair_src"]) for r in dataset.records)
        print(f"L_trans: {pairs:,} transition pairs over {len(dataset)} samples")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=(
            getattr(torch, config["model_dtype"]) if config.get("model_dtype")
            else torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ),
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    print(f"model weights dtype: {next(model.parameters()).dtype}")

    if config.get("use_lora"):
        target_modules = config.get(
            "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ).split(",")
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        # gradient checkpointing backprops through a frozen base model here; without this the
        # graph has no leaf requiring grad and checkpointed activations produce no gradient.
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        print(f"trainable param dtype: {next(p for p in model.parameters() if p.requires_grad).dtype}")

    trainer_cls = MaskedSFTTrainer
    trainer_kwargs = {
        "objective": config.get("objective") or "nll",
        "prefix_objective": config.get("prefix_objective"),
        "cont_objective": config.get("cont_objective"),
        "junction_objective": config.get("junction_objective"),
    }
    print(
        f"objective: {trainer_kwargs['objective']} (prefix override: {trainer_kwargs['prefix_objective']}, "
        f"continuation override: {trainer_kwargs['cont_objective']}, "
        f"junction override: {trainer_kwargs['junction_objective']})"
    )
    if use_trans:
        num_layers = len(find_decoder_layers(model))
        config.setdefault("trans_layer", default_transition_layer(num_layers))
        config.setdefault("trans_hidden", 1024)
        config.setdefault("trans_lr", 2e-4)
        config.setdefault("trans_lambda_warmup", True)
        config.setdefault("trans_shuffle_targets", False)
        config.setdefault("trans_grad_log_interval", 50)
        predictor = attach_predictor(model, config["trans_hidden"])
        print(
            f"L_trans: lambda={config['trans_lambda']} layer={config['trans_layer']}/{num_layers} "
            f"predictor {sum(p.numel() for p in predictor.parameters()):,} params lr={config['trans_lr']} "
            f"warmup={config['trans_lambda_warmup']} shuffle_targets={config['trans_shuffle_targets']}"
        )
        trainer_cls = TransitionSFTTrainer
        trainer_kwargs |= {
            key: config[key]
            for key in (
                "trans_lambda", "trans_layer", "trans_lr", "trans_lambda_warmup",
                "trans_shuffle_targets", "trans_grad_log_interval",
            )
        }

    if baseline == "asft":
        reference = AutoModelForCausalLM.from_pretrained(
            config["model_name"], dtype=next(model.parameters()).dtype,
            attn_implementation=config.get("attn_implementation", "sdpa"),
        )
        reference.config.use_cache = False
        trainer_cls = AnchoredSFTTrainer
        trainer_kwargs |= {"reference_model": reference, "kl_lambda": config.get("asft_kl_lambda", 0.05)}
        print(f"ASFT: kl_lambda={trainer_kwargs['kl_lambda']}, reference = frozen pretrained model")
    elif baseline == "vcore":
        trainer_cls = VCoreSFTTrainer
        trainer_kwargs |= {
            "probe_epsilon": config.get("vcore_epsilon", 1e-4),
            "temperature": config.get("vcore_temperature", 5000.0),
            "probe_batches": config.get("vcore_probe_batches"),
        }
        print(f"VCORE: epsilon={trainer_kwargs['probe_epsilon']} temperature={trainer_kwargs['temperature']}")

    callbacks = []
    step_logger = None
    if config.get("run_log_dir"):
        if int(os.environ.get("WORLD_SIZE", 1)) > 1 or baseline or use_trans:
            parser.error("--run-log-dir covers single-GPU masked SFT only")
        collator = MaskedSFTCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        selection = bool(config.get("selection_log"))
        step_logger = (selection_log.SelectionStepLogger(config["run_log_dir"], dataset) if selection
                       else StepLogger(config["run_log_dir"]))
        callbacks.append(step_logger)
        config["logging_steps"] = 1
        if selection:
            selection_log.data_check(dataset, collator, Path(config["run_log_dir"]))
        else:
            data_check(dataset, collator, tokenizer, Path(config["run_log_dir"]))
        if config.get("param_check_step"):
            callbacks.append(ParamChangeCheck(config["run_log_dir"], config["param_check_step"]))
        if config.get("epoch_nll") or config.get("epoch_checkpoints_dir"):
            epoch_end = selection_log.SelectionEpochEnd if selection else EpochEnd
            callbacks.append(epoch_end(config["run_log_dir"], dataset, tokenizer, step_logger,
                                      config.get("epoch_checkpoints_dir"), bool(config.get("epoch_nll")),
                                      int(config["epochs"])))
    elif any(config.get(k) for k in ("epoch_nll", "epoch_checkpoints_dir", "param_check_step", "selection_log")):
        parser.error("--epoch-nll, --epoch-checkpoints-dir, --param-check-step and --selection-log need --run-log-dir")

    trainer = trainer_cls(
        model=model,
        args=build_training_arguments(config),
        train_dataset=dataset,
        data_collator=MaskedSFTCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
        # lets Trainer write tokenizer files into every checkpoint-N/, not just output_dir at the
        # end — otherwise intermediate checkpoints aren't loadable by vLLM (sgl.eval.evaluate) on their own.
        processing_class=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    if step_logger is not None:
        trainer.step_logger = step_logger
        write_run_config(Path(config["run_log_dir"]), config, trainer.args, model)

    # --resume: pick up the last checkpoint in output_dir if one is present (a re-run after a crash),
    # otherwise start fresh. Guarded so a first run with an empty/absent output_dir doesn't error.
    resume_from = None
    if config.get("resume"):
        from transformers.trainer_utils import get_last_checkpoint

        out_dir = config["output_dir"]
        last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
        if last:
            print(f"--resume: continuing from {last}")
            resume_from = last
            if use_trans:
                # DeepSpeed's checkpoint restores it too (trainable submodule); this covers the
                # non-DeepSpeed path and any checkpoint whose engine state was pruned.
                print(f"--resume: predictor {'loaded' if trainer.load_predictor(last) else 'NOT found'} in {last}")
        else:
            print("--resume: no existing checkpoint found, starting fresh")

    result = trainer.train(resume_from_checkpoint=resume_from)

    # Persist every logged metric (per-step loss/grad_norm/lr/epoch + the final train_* summary)
    # to a standalone JSON so the full training curve survives independently of the stdout log.
    if config.get("metrics_log"):
        metrics_path = Path(config["metrics_log"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "config": {
                        "model_name": config["model_name"],
                        "data_path": config["data_path"],
                        "epochs": config["epochs"],
                        "learning_rate": config["learning_rate"],
                        "min_learning_rate": config.get("min_learning_rate"),
                        "per_device_batch_size": config["per_device_batch_size"],
                        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                        "effective_batch_size": config["per_device_batch_size"]
                        * config["gradient_accumulation_steps"],
                        "warmup_ratio": config.get("warmup_ratio"),
                        "seed": config.get("seed", 42),
                        "use_lora": bool(config.get("use_lora")),
                        "trans_lambda": config.get("trans_lambda"),
                        "trans_layer": config.get("trans_layer"),
                        "trans_shuffle_targets": config.get("trans_shuffle_targets"),
                    },
                    "final_metrics": result.metrics,
                    "log_history": trainer.state.log_history,
                },
                indent=2,
            )
        )
        print(f"metric history ({len(trainer.state.log_history)} entries) -> {metrics_path}")

    if config.get("use_lora") and config.get("lora_merge", True):
        # merge adapters into the base weights so output_dir is a normal full checkpoint,
        # loadable by sgl.eval.evaluate/vLLM exactly like a non-LoRA run.
        trainer.model.merge_and_unload().save_pretrained(config["output_dir"])
    else:
        # --no-lora-merge: trainer.save_model() on a PEFT model saves adapter weights only
        # (tens-hundreds of MB) instead of a full ~16GB checkpoint; sgl.eval.evaluate loads it as a
        # LoRA adapter on top of the base model.
        trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    if use_trans:
        # Not needed at inference (sgl.eval.evaluate/vLLM read only the adapter); kept for continued training.
        trainer.save_predictor(config["output_dir"])

    if step_logger is not None:
        step_logger.write_summary(config["output_dir"], gpus=trainer.args.world_size)

    # evidence for the paper's "fewer supervised tokens" claim, recorded next to the checkpoint
    (Path(config["output_dir"]) / "run-summary.json").write_text(
        json.dumps(
            {
                "data_path": config["data_path"],
                "samples": len(dataset),
                "supervised_tokens": dataset.supervised_token_count(),
                "epochs": config["epochs"],
                "seed": config.get("seed", 42),
                "train_runtime_s": result.metrics.get("train_runtime"),
                "final_train_loss": result.metrics.get("train_loss"),
                "trans_lambda": config.get("trans_lambda"),
                "trans_pairs": sum(len(r["pair_src"]) for r in dataset.records) if use_trans else None,
                **({"method": trainer.method_summary()} if baseline else {}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
