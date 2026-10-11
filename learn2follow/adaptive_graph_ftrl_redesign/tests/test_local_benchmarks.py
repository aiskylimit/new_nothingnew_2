"""Regression checks for the offline OPSD-compatible benchmark path."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.prepare_data import EVAL_DATASETS, prepare_eval_dataset
from adaptive_graph_ftrl_redesign.benchmark_partition import benchmark_question_ids
from adaptive_graph_ftrl_redesign.eval_adapter import eval_command


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    def no_hub(*args, **kwargs):
        raise AssertionError("Benchmark loading must not call the Hub")

    monkeypatch.setattr(datasets, "load_dataset", no_hub)
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    for name in EVAL_DATASETS:
        source = raw / "eval" / name / "data"
        source.mkdir(parents=True)
        pq.write_table(pa.table({"problem": [f"question {name}"], "answer": [7], "id": [123]}),
                       source / "train.parquet")
        # The default data/ configuration must take precedence over alternatives.
        pq.write_table(pa.table({"problem": ["duplicate"], "answer": [0]}),
                       source.parent / "alternative.parquet")
        prepare_eval_dataset(name, raw, processed)
    monkeypatch.setenv("EVAL_DATASET_DIR", str(processed / "eval"))
    return processed / "eval"


def test_local_preparation_and_training_exclusion(prepared):
    result = benchmark_question_ids(EVAL_DATASETS)
    assert len(result["benchmark_question_ids"]) == 3
    assert [source["task"] for source in result["benchmark_sources"]] == list(EVAL_DATASETS)
    for name in EVAL_DATASETS:
        rows = datasets.load_from_disk(str(prepared / name))
        assert rows[0] == {"problem_id": 123, "problem": f"question {name}", "answer": "7"}
        assert len(rows) == 1


@pytest.mark.parametrize("launcher", ["python", "se-opsd"])
def test_eval_command_forwards_local_dataset_dir(prepared, launcher):
    cmd, _ = eval_command("/run/checkpoint-100", "hmmt25", gpus="2,3",
                          dataset_dir=str(prepared), launcher=launcher)
    assert cmd[cmd.index("--dataset_dir") + 1] == str(prepared)


def test_evaluator_reads_normalized_local_rows(prepared, monkeypatch):
    # GPU libraries are irrelevant to checking local loading and prompt creation.
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=object, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object))
    monkeypatch.setitem(sys.modules, "math_verify", SimpleNamespace(parse=None, verify=None))
    path = Path(__file__).resolve().parents[2] / "eval" / "evaluate_math.py"
    spec = importlib.util.spec_from_file_location("local_math_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    class GenerationReached(Exception):
        pass

    def generate(prompts, params, **kwargs):
        assert len(prompts) == 1 and "question hmmt25" in prompts[0]
        assert params.n == 12
        raise GenerationReached

    llm = SimpleNamespace(
        generate=generate,
        llm_engine=SimpleNamespace(
            model_config=SimpleNamespace(dtype="bfloat16", quantization=None),
            cache_config=SimpleNamespace(cache_dtype="auto")))
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages[0]["content"])
    with pytest.raises(GenerationReached):
        evaluator.evaluate_math500(llm, tokenizer, 38912, dataset_name="hmmt25", val_n=12,
                                   dataset_dir=str(prepared))
