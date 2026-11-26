"""Segment-Selective SFT ports."""
import json
import random
from pathlib import Path

import pytest

from sgl.data.s1k import convert_rows, last_boxed
from sgl.segment.rules import split_segments
from sgl.ssft import attribution, build

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))


@pytest.fixture(scope="module")
def rows():
    rng = random.Random(0)
    out = []
    for line in (ASSETS / "palign_sample.jsonl").open():
        row = json.loads(line)
        solution = row["output"]
        segments = split_segments(solution, "paragraph")
        out.append({
            "question": row["input"] + "  ", "solution": solution, "segments": segments,
            "answer": last_boxed(solution) or "0",
            "selected_spans_ids": sorted(rng.sample(range(len(segments)), k=min(3, len(segments)))),
        })
    return out


# ------------------------------------------------------------------ data


def test_last_boxed_and_row_conversion():
    assert last_boxed("x \\boxed{1} y \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    source = [
        {"question": " q1 ", "deepseek_thinking_trajectory": "t \\boxed{5}", "solution": "5"},
        {"question": "q2", "deepseek_thinking_trajectory": "", "solution": "1"},
        {"question": "q3", "deepseek_thinking_trajectory": "no box", "solution": ""},
        {"question": "q4", "deepseek_thinking_trajectory": "\\boxed{" + "9" * 300 + "}", "solution": ""},
        {"question": "q5", "deepseek_thinking_trajectory": "\\boxed{2}", "solution": "3"},
    ]
    rows, stats = convert_rows(source, "deepseek_thinking_trajectory", "trace", 200)
    assert [r["question"] for r in rows] == ["q1", "q5"]
    assert [r["answer"] for r in rows] == ["5", "2"]
    assert stats == {"no_trace": 1, "no_answer": 1, "answer_too_long": 1, "trace_disagrees_with_gt": 1}
    rows, _ = convert_rows(source, "deepseek_thinking_trajectory", "gt", 200)
    assert [r["answer"] for r in rows] == ["5", "3"]


# ------------------------------------------------------------------ build (train_mask.py labels)


def test_build_rejects_special_think_prefix_and_bad_templates(tokenizer):
    with pytest.raises(ValueError):
        build.template_mode(False, "special")
    chat_kwargs, think, expected = build.template_mode(True, "off")  # R1 ending, but a ChatML template
    with pytest.raises(SystemExit):
        build.check_template(tokenizer, chat_kwargs, think, expected)


# ------------------------------------------------------------------ attribution (Integrated Gradients)


def test_build_example_spans_and_answer_range(tokenizer, rows):
    # Upstream locates the answer by scanning token strings for "boxed", "{" and "}", which assumes
    # "{" and the closing "}" are separate tokens (true for Qwen's tokenizer). Kept as is for
    # fidelity; a plain numeric answer keeps this check tokenizer independent.
    row = dict(rows[1], answer="42")
    ids, spans, (start, end) = attribution.build_example(tokenizer, row)
    nonempty = [span for span in spans if span != (0, 0)]
    assert tokenizer.decode(ids[nonempty[0][0]:nonempty[-1][1]]) == "".join(row["segments"])
    assert all(a < b for a, b in nonempty) and all(b == c for (_, b), (c, _) in zip(nonempty, nonempty[1:]))
    assert tokenizer.decode(ids[start:end]) == "42"


def test_attribution_prompt_ids_are_the_rendered_chat_template(tokenizer, rows):
    """Upstream ran transformers 4.x, where apply_chat_template(tokenize=True) returns the ids of the
    rendered template without extra special tokens; transformers 5 returns a BatchEncoding instead."""
    row = dict(rows[0], answer="7")
    ids, spans, _ = attribution.build_example(tokenizer, row)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": attribution.USER_TEMPLATE.format(input=row["question"])}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids[:len(prompt_ids)] == prompt_ids
    assert spans[0][0] == len(prompt_ids)
