from dataclasses import replace
import numpy as np
import pytest

from adaptive_graph_ftrl_redesign.data import load_examples, parse_example, render_mask, terminal_span
from adaptive_graph_ftrl_redesign.masks import candidate_pool, Pool
from .fixtures import row, ByteTokenizer


def test_split_mixed_step_nested_and_intermediate_box():
    original = row(
        fragments=[
            "Known \\boxed{4}.\n",
            "Compute x=7.\n",
            "Use x to conclude. Final answer: \\boxed{\\frac{1}{2}}",
        ]
    )
    example = parse_example(original, 0)
    assert example.gold_answer == r"\frac{1}{2}"
    assert example.reasoning_source + example.final_answer_block == example.reference_solution
    assert "Use x to conclude." in example.nodes[-1].text
    assert r"\boxed{4}" in example.nodes[0].text
    assert not any(n.protected for n in example.nodes)
    original["answer"] = "gold-label-format"
    assert parse_example(original, 0).gold_answer == "gold-label-format"


def test_answer_only_removed_outgoing_rejected_and_flags():
    original = row(fragments=["Given x=1.\n", "Use y=2.\n", r"Final answer: \boxed{3}"])
    ex = parse_example(original, 0)
    assert ex.node_ids == (0, 1)
    assert set(ex.parents_by_child) == {1}
    all_nodes = frozenset(ex.node_ids)
    assert render_mask(ex, all_nodes, True).strip() == ex.final_answer_block
    assert render_mask(ex, all_nodes, False) == ""
    original["hypergraph"]["hyperedges"].append({"parent_steps": [2], "target_step": 0})
    with pytest.raises(ValueError, match="outgoing"):
        parse_example(original, 0)


def test_invalid_terminal_answer_and_empty_graph():
    with pytest.raises(ValueError, match="reliable"):
        terminal_span({}, "An intermediate \\boxed{2}. Continue deriving x.")
    with pytest.raises(ValueError, match="malformed"):
        terminal_span({}, r"Final answer: \boxed{\frac{1}{2}")
    with pytest.raises(ValueError, match="no reasoning"):
        parse_example(row(fragments=[r"Final answer: \boxed{3}"], edges=[]), 0)
    source = "Compute 2. Answer: 2"
    assert source[slice(*terminal_span({"answer_span": [11, len(source)]}, source))] == "Answer: 2"


def test_mask_answer_has_explicit_label_without_changing_contents():
    ex = parse_example(row(), 0)
    ex = replace(ex, final_answer_block=r"\boxed{\frac{1}{2}}")
    masked = render_mask(ex, frozenset(ex.node_ids), True)
    assert masked == "Final answer:\n" + ex.final_answer_block
    assert render_mask(ex, frozenset(), False) == "\n\n".join(n.text for n in ex.nodes)
    labeled = replace(ex, final_answer_block=r"Final answer: \boxed{3}")
    assert render_mask(labeled, frozenset(ex.node_ids), True) == labeled.final_answer_block


def test_chain_fork_pool_and_pair_limit():
    ex = parse_example(row(), 0)
    pool = candidate_pool(ex, ByteTokenizer())
    masks = [m for values in pool.groups.values() for m in values]
    assert frozenset({0, 1, 2}) in masks and frozenset({1, 2}) in masks
    assert frozenset({0}) not in masks
    assert len(masks) == len(set(masks))
    fork = replace(ex, parents_by_child={1: (0,), 2: (0,)})
    fork_pool = candidate_pool(fork, ByteTokenizer(), pair_limit=1)
    assert any(frozenset({1}) == m for group in fork_pool.groups.values() for m in group)
    assert fork_pool.pair_used == 1 and fork_pool.pair_total == 3
    assert candidate_pool(fork, ByteTokenizer(), pair_limit=1).groups == fork_pool.groups
    prefix = candidate_pool(fork, ByteTokenizer(), variant="prefix")
    assert all(
        tuple(sorted(set(fork.node_ids) - m)) in [(), (0,), (0, 1)]
        for group in prefix.groups.values()
        for m in group
    )


def test_unique_level_not_candidate_count_bias():
    pool = Pool({0: [frozenset({i}) for i in range(10)], 10: [frozenset({10})]}, 10, {}, 0, 0)
    rng = np.random.default_rng(42)
    values = [pool.sample(0.5, 0.05, rng)[1] for _ in range(5000)]
    assert 0.47 < np.mean(values) < 0.53


def test_malformed_records_are_explicitly_excluded(tmp_path):
    import json

    malformed = row()
    malformed["hypergraph"]["semantic_steps"][0] = None
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [[], malformed, row()]) + "\n")
    examples, metadata = load_examples(path)
    assert len(examples) == 1 and metadata["excluded"] == 2
    assert metadata["invalid_rows"][0]["answer_split_valid"] is False
    assert metadata["invalid_rows"][1]["answer_split_valid"] is True
