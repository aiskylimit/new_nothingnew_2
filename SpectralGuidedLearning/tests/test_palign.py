"""P-ALIGN ports, checked against its published labels and released data."""
import gzip
import json
from pathlib import Path

import pytest

from sgl.eval.graders import palign as palign_grader
from sgl.palign import build, truncate

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
RELEASED = Path(__file__).resolve().parents[1] / "references" / "P-ALIGN" / "data" / "palign_sft_qwen2.5-7b.json.gz"


@pytest.mark.parametrize("response,verdict", [
    ("[ENOUGH]", True), ("  ENOUGH \n", True), ("[NOT_ENOUGH]", False), ("NOT ENOUGH", False),
    ("ENOUGH.", False), ("I think [ENOUGH] here", True), ("ERROR: boom", False),
])
def test_verdict_rule(response, verdict):
    assert truncate.is_sufficient(response) == verdict


def test_grader_reproduces_palign_published_labels():
    rows = [json.loads(line) for line in (ASSETS / "palign_labels.jsonl").open()]
    assert len(rows) == 60
    for row in rows:
        assert palign_grader.grade_math_verify(row["outputs"], row["answer"]) == row["labels"]


def test_assemble_reproduces_the_released_marker_layout():
    released = json.load(gzip.open(RELEASED))
    assert len(released) == 966
    for row in released:
        head, continuation = row["output"].split(build.END + "\n", 1)
        assert head.startswith(build.BEGIN)
        prefix = head[len(build.BEGIN):]
        assert build.assemble(prefix, continuation) == row["output"]
        assert row["instruction"] == build.INSTRUCTION


def test_build_filters_by_answer(tmp_path):
    truncated = tmp_path / "t.jsonl"
    aligned = tmp_path / "a.jsonl"
    truncated.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "q1", "answer": "4"}, {"question": "q2", "answer": "5"}, {"question": "q3", "answer": ""}]))
    aligned.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "q1", "sufficient_reasoning": "Two plus two.", "output": " so \\boxed{4}\n"},
        {"question": "q2", "sufficient_reasoning": "x.", "output": "\\boxed{6}"},
        {"question": "q3", "sufficient_reasoning": "y.", "output": "\\boxed{1}"}]))
    out = tmp_path / "data.json"
    build.main(["--aligned", str(aligned), "--truncated", str(truncated), "--output", str(out)])
    rows = json.loads(out.read_text())
    assert [r["input"] for r in rows] == ["q1"]
    assert rows[0]["output"] == "<Begin_of_Prefix>Two plus two.<End_of_Prefix>\nso \\boxed{4}"
    build.main(["--aligned", str(aligned), "--truncated", str(truncated), "--output", str(out), "--keep-unverified"])
    assert len(json.loads(out.read_text())) == 3


def test_truncate_cli_with_the_hf_backend(tmp_path):
    """Smoke: a random tiny judge never says [ENOUGH], so every trace keeps all its sentences."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))
    torch.manual_seed(0)
    model_dir = tmp_path / "judge"
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    source = tmp_path / "traces.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in [
        {"question": "q1", "solution": "First. Second. Third.", "answer": "1"},
        {"question": "", "solution": "skipped: no question", "answer": "2"},
        {"question": "q3", "solution": "Only one", "answer": "3"},
    ]))
    out = tmp_path / "truncated.jsonl"
    truncate.main(["--model", str(model_dir), "--input", str(source), "--output", str(out),
                   "--backend", "hf", "--max-new-tokens", "3"])
    rows = [json.loads(line) for line in out.open()]
    assert [r["id"] for r in rows] == [0, 2]
    for row in rows:
        if not row["is_sufficient"]:
            assert row["sufficient_sentences"] == row["total_sentences"] and row["prefix_ratio"] == 1.0
