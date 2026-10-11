from dataclasses import asdict, dataclass, fields
import math

import yaml


@dataclass
class Config:
    method_version: str = "graph_virtual_adam_atesd_v1"
    model: str = "Qwen/Qwen3-4B"
    dataset: str = "graph_based_outputs/hypergraph_openthoughts30k_operations_v3.jsonl"
    output_dir: str = "adaptive_graph_ftrl_redesign/outputs/graph_answer"
    variant: str = "graph"
    mask_teacher_include_final_answer: bool = True
    frozen_controller: bool = False
    fixed_mixture: bool = False
    teacher_selection: str = "soft"
    seed: int = 42
    total_updates: int = 100
    full_opsd_warmup_steps: int = 5
    window_steps: int = 5
    effective_batch_size: int = 32
    gradient_accumulation_steps: int = 8
    microbatch_size: int = 1
    control_pool_size: int = 128
    control_batch_size: int = 16
    min_valid_controls: int = 8
    direct_instruction: str = "Return only the final answer, without reasoning, inside \\boxed{}."
    direct_answer_cue: str = "\nFinal answer: \\boxed{"
    warmup_windows: int = 3
    exposure_temperature: float = 0.05
    max_pair_candidates: int = 4096
    controller_lr: float = 0.01
    controller_max_grad_norm: float = 1.0
    gamma_discount: float = 0.95
    lambda_gt: float = 0.01
    grounded_score_examples_per_step: int = 4
    grounded_reference_max_tokens: int = 1024
    grounded_reference_span: str = "prefix_then_answer"
    ftrl_rho: float = 0.9
    ftrl_eta: float = 3.0
    ftrl_min_weight: float = 0.02
    delta_rms_decay: float = 0.9
    delta_rms_floor: float = 1e-4
    delta_scaled_clip: float = 1.0
    normalizer_decay: float = 0.9
    reward_variance_floor: float = 1e-4
    delta_variance_floor: float = 1e-8
    standardizer_clip: float = 5.0
    verbose_runtime: bool = False
    answer_temperature: float = 1.0
    numeric_epsilon: float = 1e-12
    main_max_new_tokens: int = 1024
    generation_temperature: float = 1.1
    top_p: float = 0.95
    distillation_temperature: float = 1.1
    generation_backend: str = "vllm"
    vllm_gpu_memory_utilization: float = 0.35
    max_length: int = 20000
    replay_cache_dir: str = ""
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 0.1
    lora_rank: int = 64
    lora_alpha: int = 128
    dtype: str = "bfloat16"
    attention_backend: str = "eager"
    gradient_checkpointing: bool = True
    chunk_size: int = 64
    teacher_thinking: bool = True
    student_thinking: bool = False
    data_limit: int = 0
    save_every: int = 25
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95
    eval_top_k: int = -1
    eval_samples: int = 12
    eval_max_new_tokens: int = 38912
    eval_seed: int = 42
    benchmark_tasks: tuple = ("aime25", "aime26", "hmmt25")

    @property
    def probe_batch_size(self):
        # Compatibility with the shared vLLM capacity setting; never generate controls.
        return 1

    def validate(self, world=1):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type is float:
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    raise ValueError(f"Invalid numeric setting: {field.name}")
                setattr(self, field.name, float(value))
            elif field.type is int and (type(value) is not int):
                raise ValueError(f"{field.name} must be an integer")
            elif field.type is bool and not isinstance(value, bool):
                raise ValueError(f"{field.name} must be a Boolean")
        if self.method_version != "graph_virtual_adam_atesd_v1":
            raise ValueError("Unsupported redesign version")
        if self.variant not in {"graph", "prefix", "full"}:
            raise ValueError("variant must be graph, prefix or full")
        if self.teacher_selection not in {"soft", "hard"}:
            raise ValueError("teacher_selection must be soft or hard")
        if self.teacher_selection == "hard" and self.fixed_mixture:
            raise ValueError("hard teacher selection cannot use fixed_mixture")
        if self.microbatch_size != 1 or self.effective_batch_size != world * self.gradient_accumulation_steps:
            raise ValueError("Batch must equal world * accumulation, microbatch=1")
        for name in (
            "total_updates",
            "window_steps",
            "effective_batch_size",
            "gradient_accumulation_steps",
            "control_pool_size",
            "control_batch_size",
            "min_valid_controls",
            "chunk_size",
            "max_length",
            "main_max_new_tokens",
            "grounded_score_examples_per_step",
            "grounded_reference_max_tokens",
            "save_every",
            "controller_lr",
            "controller_max_grad_norm",
            "exposure_temperature",
            "ftrl_eta",
            "delta_rms_floor",
            "delta_scaled_clip",
            "reward_variance_floor",
            "delta_variance_floor",
            "standardizer_clip",
            "numeric_epsilon",
            "learning_rate",
            "adam_epsilon",
            "answer_temperature",
            "generation_temperature",
            "distillation_temperature",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.full_opsd_warmup_steps <= self.total_updates or self.warmup_windows < 0:
            raise ValueError("Invalid FULL/actor warmup")
        if not self.min_valid_controls <= self.control_batch_size <= self.control_pool_size:
            raise ValueError("Invalid control batch/threshold/pool")
        if self.grounded_score_examples_per_step > self.effective_batch_size:
            raise ValueError("Grounded subset exceeds main batch")
        if (
            self.max_pair_candidates < 0
            or self.lambda_gt < 0
            or self.max_grad_norm < 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid pair cap/coefficient/clipping/decay")
        if not 0 <= self.ftrl_min_weight < 0.5 or not 0 <= self.ftrl_rho <= 1:
            raise ValueError("Invalid FTRL floor/decay")
        if any(
            not 0 <= x < 1
            for x in (self.normalizer_decay, self.delta_rms_decay, self.adam_beta1, self.adam_beta2)
        ):
            raise ValueError("Invalid EMA/Adam decay")
        if not 0 < self.gamma_discount <= 1 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid discount/top-p")
        if (
            self.generation_backend not in {"vllm", "transformers"}
            or not 0 < self.vllm_gpu_memory_utilization < 1
        ):
            raise ValueError("Invalid generation settings")
        if not self.teacher_thinking or self.student_thinking:
            raise ValueError("Keep teacher thinking and student non-thinking")
        if (
            not self.direct_answer_cue
            or not self.direct_instruction
            or self.grounded_reference_span != "prefix_then_answer"
        ):
            raise ValueError("Invalid direct template/reference span")
        return self


def load_config(path, overrides=None):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return Config(**data)


def config_dict(config):
    return asdict(config)
