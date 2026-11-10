# Math and code evaluation

Provision the `contra` environment from [contra.txt](../../contra.txt). It uses
LM Evaluation Harness 0.4.11 with vLLM 0.11.0. Run the commands below from the
`contra-kd-math-code` directory.

## Checkpoints

List the 16 destination directories before downloading the checkpoints.

```bash
bash project_commands.sh --list
```

GPT-2 entries are full model directories. OpenLLaMA2 entries are LoRA adapter
directories with `adapter_config.json` and `adapter_model.safetensors` or
`adapter_model.bin`. Put the shared `openlm-research/open_llama_3b_v2` base model,
including its tokenizer, in `checkpoints/openllama2-3B`. Set
`OPENLLAMA2_BASE_MODEL` to use a different local directory. Include any tokenizer
files saved with each adapter. The launcher uses them when present and otherwise
uses the base tokenizer.

GPT-2 +Contra defaults to the sampled-time launcher's `contra_sampled/` layout.
Set `GPT2_CONTRA_DIR=contra` for checkpoints saved to the lower-level training
scripts' `contra/` layout.

## Offline benchmark data

On a connected machine, populate the cache using the same provisioned environment.
The task loader caches the datasets and the Hugging Face `code_eval` metric.
Copy the resulting cache to the offline machine and use the same `HF_HOME` there.
Keep HumanEval and MBPP runs in an isolated environment because their scorers
execute generated Python.

```bash
source /mnt/local/uvenvs/contra/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 HF_ALLOW_CODE_EVAL=1 python - <<'PY'
from lm_eval.tasks import get_task_dict

get_task_dict(["gsm8k_cot", "hendrycks_math", "hendrycks_math500", "humaneval", "mbpp"])
PY
```

## Run evaluations

Smoke-test one GPT-2 checkpoint with five examples per task and the MATH-500 subset.

```bash
CUDA_VISIBLE_DEVICES=4 LIMIT=5 TASKS=gsm8k_cot,hendrycks_math500,humaneval,mbpp \
    bash project_commands.sh . gpt2 distillm baseline
```

Run the full suite over all 16 checkpoints on GPUs 4, 5, 6, and 7.

```bash
bash project_commands.sh
```

The positional selectors after the project directory are family
(`gpt2`, `openllama2`, or `all`), method (`distillm`, `distillm2`, `csd`,
`fdd`, or `all`), and variant (`baseline`, `contra`, or `all`). Each defaults
to `all`. Each GPU evaluates one checkpoint at a time. Set
`CUDA_VISIBLE_DEVICES` to use a different GPU list.

The full suite uses GSM8K CoT, all seven subjects of MATH, HumanEval, and the
MBPP full configuration's test split. It uses task-default prompts and greedy
decoding, with a common 256-token generation cap to leave prompt room on GPT-2.
`MAX_GEN_TOKS` overrides that cap. Use the same settings across checkpoints.
The harness saves scores, per-example outputs, and an `eval.log` per checkpoint
under `results/eval_math_code/`. Limited runs go under its `smoke/` subdirectory.
A failed checkpoint is reported with its log path. The other evaluations finish
and the launcher returns a nonzero status.

## OpenLLaMA2 LoRA

The default vLLM backend uses the shared base model plus each adapter through
`lora_local_path`. The default maximum rank is 16, matching the training
scripts. Set `MAX_LORA_RANK` for a higher-rank adapter. Set `EVAL_BACKEND=hf`
to run the same benchmarks with the Transformers/PEFT backend.

If a particular adapter needs a merged checkpoint for vLLM, export it to a
separate directory with the existing CPU merge tool. This example exports
DistiLLM +Contra and uses the same tokenizer as the direct adapter run.

```bash
BASE_MODEL="${OPENLLAMA2_BASE_MODEL:-$PWD/checkpoints/openllama2-3B}"
ADAPTER="$PWD/results/openllama2/train/contra/distillm/3B_7B"
MERGED_ROOT="$PWD/results/openllama2/merged"
TOKENIZER="$BASE_MODEL"
if [[ -f "$ADAPTER/tokenizer.json" || -f "$ADAPTER/tokenizer.model" ]]; then
    TOKENIZER="$ADAPTER"
fi

python tools/merge_lora.py --base-model "$BASE_MODEL" --lora "$ADAPTER" \
    --tokenizer "$TOKENIZER" --output-dir "$MERGED_ROOT/contra/distillm/3B_7B"

CUDA_VISIBLE_DEVICES=4 OPENLLAMA2_MERGED_ROOT="$MERGED_ROOT" \
    bash project_commands.sh . openllama2 distillm contra
```

Use the same path relative to `results/openllama2/train/` for each merged export
under `MERGED_ROOT`. Merged evaluations write to the `openllama2-merged/` output
subdirectory.
