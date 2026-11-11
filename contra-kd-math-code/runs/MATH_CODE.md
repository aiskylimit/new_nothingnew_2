# Math and code evaluation

Provision the `contra` environment from [contra.txt](../../contra.txt). It uses
LM Evaluation Harness 0.4.11 with vLLM 0.11.0. Run the commands below from the
`contra-kd-math-code` directory.

## Checkpoints

List the 22 checkpoint paths for the suite.

```bash
bash project_commands.sh --list
```

`download.txt` provisions the 16 distillation checkpoints and four pretrained
models under `checkpoints/` (`gpt2-base`, `gpt2-xlarge`, `openllama2-3B`, and
`openllama2-7B`). The GPT-2 student SFT script exports a full model to
`results/gpt2/train/sft/gpt2-base`. The OpenLLaMA2 student SFT script exports a
LoRA adapter to `results/openllama2/train/sft/openllama2-3B`.

All OpenLLaMA2 adapters use the shared `checkpoints/openllama2-3B` base.
Set `OPENLLAMA2_BASE_MODEL` to use another local directory for that base.
The launcher uses the adapter's tokenizer when present and otherwise uses the
base tokenizer.

GPT-2 +Contra defaults to the sampled-time launcher's `contra_sampled/` layout.
Set `GPT2_CONTRA_DIR=contra` for checkpoints saved to the lower-level training
scripts' `contra/` layout.

## Offline benchmark data

Run the downloader with `download.txt` to provision GSM8K, the seven MATH
subjects, MATH-500, HumanEval, and MBPP. The downloader places them under
`/mnt/local/@PROJECT@/contra-kd-math-code/data/eval/`. The launcher uses
`data/eval/` under the selected project directory by default.
Set `EVAL_DATA_DIR` to the downloaded directory if the code is elsewhere.
`HF_HOME` controls the cache for processed data.

The project includes the [Hugging Face Evaluate v0.4.6 scorer](https://github.com/huggingface/evaluate/tree/v0.4.6/metrics/code_eval)
and its `execute.py` helper under `tools/code_eval/`.

The launcher checks that the selected tasks load from the downloaded files
before starting GPU jobs. You can also run this check directly with the
`contra` environment active.

```bash
python3 tools/local_lm_eval.py check --tasks gsm8k_cot,hendrycks_math,hendrycks_math500,humaneval,mbpp
```

## Run evaluations

Smoke-test one GPT-2 checkpoint with five examples per task and the MATH-500 subset.

```bash
CUDA_VISIBLE_DEVICES=4 LIMIT=5 TASKS=gsm8k_cot,hendrycks_math500,humaneval,mbpp \
    bash project_commands.sh . gpt2 distillm baseline
```

Run student SFT and then the full benchmark suite.

```bash
bash project_commands.sh
```

The launcher trains GPT-2 and OpenLLaMA2 on the first two configured GPUs
(4 and 5 by default) and waits for both checkpoints before evaluating all 22
models on GPUs 4 and 5. With one GPU configured, the SFT jobs run
sequentially. Completed student checkpoints are reused. Training logs are
under `logs/student_sft/`. If the processed Dolly data is missing, the launcher
prepares it from the sibling `contra-kd/data/raw/dolly` download. Set
`DOLLY_DATA_DIR` if you provisioned the raw data elsewhere.

To run just student SFT on GPUs 4 and 5, use the training-only mode.

```bash
CUDA_VISIBLE_DEVICES=4,5 bash project_commands.sh --sft-only
```

For a single student, pass the project directory and family, such as
`bash project_commands.sh --sft-only . gpt2`. The training-only mode uses the
two pretrained student bases and Dolly data.

Pass a project directory to run evaluations against existing checkpoints, or
add `--train-sft` before the directory to train the selected students first.

```bash
bash project_commands.sh .
bash project_commands.sh --train-sft . openllama2 sft
```

The positional selectors after the project directory are family
(`gpt2`, `openllama2`, or `all`), method (`distillm`, `distillm2`, `csd`,
`fdd`, `sft`, `pretrained`, or `all`), and variant (`baseline`, `contra`,
or `all`). Each defaults to `all`. Use `baseline` for SFT and pretrained
checkpoints. Each GPU evaluates one checkpoint at a time. Set
`CUDA_VISIBLE_DEVICES` to use a different GPU list.

The full suite uses GSM8K CoT, all seven subjects of MATH, HumanEval, and the
MBPP full configuration's test split. It uses task-default prompts and greedy
decoding, with a common 256-token generation cap to leave prompt room on GPT-2.
`MAX_GEN_TOKS` overrides that cap. Use the same settings across checkpoints.
The harness saves scores, per-example outputs, and an `eval.log` per checkpoint
under `results/eval_math_code/`. Limited runs go under its `smoke/` subdirectory.
A failed checkpoint is reported with its log path. The other evaluations finish
and the launcher returns a nonzero status.

## Export results

```bash
bash project_commands.sh export
```

The export copies `results_*.json` and `eval.log` files, including logs from
failed runs, to `results/eval_math_code_export/` with the original relative
paths. Running it again replaces the previous export. If the evaluations used a
custom `EVAL_OUTPUT_DIR`, set it for the export as well or pass the directory as
`bash project_commands.sh export /path/to/evaluations`. The export then goes to
the sibling directory with `_export` appended to the source directory's name.

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
