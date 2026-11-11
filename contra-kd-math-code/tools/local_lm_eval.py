"""Load the math and code benchmark snapshots provisioned by download.txt."""

import argparse
import os
import sys
from pathlib import Path


PARQUET_DATASETS = {
    "openai/gsm8k": ("gsm8k", "main"),
    "EleutherAI/hendrycks_math": ("hendrycks_math", None),
    "openai/openai_humaneval": ("humaneval", "openai_humaneval"),
    "google-research-datasets/mbpp": ("mbpp", "full"),
}


def install_local_loaders(data_root):
    import datasets
    import evaluate

    load_dataset = datasets.load_dataset
    load_metric = evaluate.load

    def load_local_dataset(path, name=None, *args, **kwargs):
        if path == "HuggingFaceH4/MATH-500":
            test_file = data_root / "MATH-500" / "test.jsonl"
            if not test_file.is_file():
                raise FileNotFoundError(f"Missing MATH-500 data at {test_file}")
            builder = "json"
            data_files = {"test": str(test_file)}
        elif path in PARQUET_DATASETS:
            directory, default_config = PARQUET_DATASETS[path]
            config_dir = data_root / directory / (name or default_config or "")
            data_files = {}
            for file in sorted(config_dir.glob("*.parquet")):
                split = file.stem.split("-", 1)[0]
                data_files.setdefault(split, []).append(str(file))
            if "test" not in data_files:
                raise FileNotFoundError(f"Missing {path} test data at {config_dir}")
            builder = "parquet"
        else:
            return load_dataset(path, name, *args, **kwargs)

        for key in ("revision", "token", "trust_remote_code"):
            kwargs.pop(key, None)
        kwargs["data_files"] = data_files
        return load_dataset(builder, name, *args, **kwargs)

    def load_local_metric(path, *args, **kwargs):
        if path == "code_eval":
            metric_dir = data_root / "code_eval"
            for filename in ("code_eval.py", "execute.py"):
                if not (metric_dir / filename).is_file():
                    raise FileNotFoundError(f"Missing code evaluator file at {metric_dir / filename}")
            path = str(metric_dir / "code_eval.py")
        return load_metric(path, *args, **kwargs)

    datasets.load_dataset = load_local_dataset
    evaluate.load = load_local_metric


def check_tasks(argv):
    from lm_eval.config.evaluate_config import EvaluatorConfig
    from lm_eval.tasks import get_task_dict

    parser = argparse.ArgumentParser(description="Check that the benchmark tasks load offline.")
    parser.add_argument("--tasks", required=True)
    args = parser.parse_args(argv)
    config = EvaluatorConfig(tasks=args.tasks)
    manager = config.process_tasks()
    get_task_dict(config.tasks, task_manager=manager)
    print(f"Loaded benchmark tasks: {args.tasks}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root / "data" / "eval"
    data_root = Path(os.environ.get("EVAL_DATA_DIR", default_data_root)).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_EVALUATE_OFFLINE", "HF_ALLOW_CODE_EVAL"):
        os.environ.setdefault(name, "1")
    install_local_loaders(data_root)

    if sys.argv[1:2] == ["check"]:
        check_tasks(sys.argv[2:])
    else:
        from lm_eval.__main__ import cli_evaluate

        cli_evaluate()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(f"{error}\nProvision the benchmark assets listed in download.txt.") from None
