"""Conservative terminal answer extraction, preserving original reference text."""

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import re

from .schema import (
    ReasoningNode,
    _boxed_span,
    _validate_acyclic,
    file_sha256,
)


PARSER_VERSION = "terminal_block_v1"
FORMATTING = re.compile(r"(?:\s|[.$*#:_!]|\\[\[\]()])*\Z")
FINAL_MARKER = re.compile(
    r"(?:\b(?:final\s+answer\s*(?:is|:)|answer\s*(?:is|:)|conclusion\s*:)|(?:Do \u0111\u00f3\s+)?\u0111\u00e1p \u00e1n cu\u1ed1i c\u00f9ng l\u00e0)",
    re.I,
)


@dataclass(frozen=True)
class Example:
    qid: str
    row_index: int
    problem: str
    reference_solution: str
    reasoning_source: str
    final_answer_block: str
    gold_answer: str
    answer_span: tuple
    nodes: tuple
    parents_by_child: dict
    raw: dict

    @property
    def node_ids(self):
        return tuple(n.node_id for n in self.nodes)


def terminal_span(graph, source):
    explicit = graph.get("final_answer_span", graph.get("answer_span"))
    if explicit is not None:
        if isinstance(explicit, dict):
            start, end = explicit["char_start"], explicit["char_end"]
        else:
            start, end = explicit
        start, end = int(start), int(end)
        if not 0 <= start < end <= len(source) or not FORMATTING.fullmatch(source[end:]):
            raise ValueError("invalid/nonterminal explicit answer span")
        return start, len(source)
    block = graph.get("final_answer_block")
    if isinstance(block, str) and block:
        start = source.rfind(block)
        if start < 0 or source.find(block) != start or not FORMATTING.fullmatch(source[start + len(block) :]):
            raise ValueError("ambiguous/nonterminal answer block")
        return start, len(source)
    box = _boxed_span(source)
    markers = list(FINAL_MARKER.finditer(source))
    if box is not None and FORMATTING.fullmatch(source[box[2] :]):
        start = box[0]
        nearby = [
            m
            for m in markers
            if m.start() < box[0]
            and len(source[m.end() : box[0]]) < 160
            and not re.search(r"[=;]", source[m.end() : box[0]])
        ]
        if nearby:
            start = nearby[-1].start()
        else:
            # Keep display delimiters with the terminal box, not as empty reasoning.
            wrapper = re.search(r"(?:\\\[|\$\$|\$)\s*\Z", source[:start])
            if wrapper:
                start = wrapper.start()
        return start, len(source)
    if "\\boxed" in source and box is None:
        raise ValueError("malformed final boxed expression")
    if markers and (box is None or markers[-1].start() > box[2]):
        marker = markers[-1]
        tail = source[marker.end() :].strip()
        if tail and len(tail) <= 256 and "\n\n" not in tail:
            return marker.start(), len(source)
    raise ValueError("no reliable terminal final-answer block")


def parse_example(row, index):
    if not isinstance(row, dict):
        raise ValueError("input record must be an object")
    graph = row.get("hypergraph")
    if isinstance(graph, str):
        graph = json.loads(graph)
    if not isinstance(graph, dict):
        raise ValueError("missing hypergraph")
    source = graph.get("solution") or graph.get("reference_solution")
    problem = graph.get("problem")
    if not isinstance(source, str) or not isinstance(problem, str) or not problem.strip():
        raise ValueError("missing solution/problem")
    start, end = terminal_span(graph, source)
    block = source[start:end]
    if "\\boxed" in block and _boxed_span(block) is None:
        raise ValueError("malformed boxed answer block")
    gold = next(
        (
            obj[key]
            for obj in (row, graph)
            for key in ("gold_answer", "answer", "final_answer")
            if isinstance(obj.get(key), (str, int, float)) and str(obj[key]).strip()
        ),
        None,
    )
    if gold is None:
        box = _boxed_span(block)
        gold = block[box[1] + 1 : box[2] - 1].strip() if box else FINAL_MARKER.sub("", block, count=1).strip()
    if not str(gold).strip():
        raise ValueError("missing gold answer label")
    steps, edges = graph.get("semantic_steps"), graph.get("hyperedges")
    if not isinstance(steps, list) or not isinstance(edges, list):
        raise ValueError("missing steps/edges")
    nodes, removed, ids = [], set(), set()
    previous_end = 0
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("semantic step must be an object")
        node = int(step["step_id"])
        text = step["text"]
        if node in ids or not isinstance(text, str) or not text.strip():
            raise ValueError("duplicate ID/empty step")
        ids.add(node)
        left, right = step.get("char_start"), step.get("char_end")
        if left is None or right is None:
            left = source.find(text)
            if left < 0 or source.find(text, left + 1) >= 0:
                raise ValueError("ambiguous step span")
            right = left + len(text)
        left, right = int(left), int(right)
        if not previous_end <= left < right <= len(source) or source[left:right].strip() != text.strip():
            raise ValueError("invalid/out-of-order step span")
        previous_end = right
        clipped = source[left : min(right, start)] if left < start else ""
        substantive = re.sub(r"^\s*(?:Step\s+)?\d+[.):]\s*", "", clipped, flags=re.I)
        if substantive.strip() and not FORMATTING.fullmatch(substantive):
            nodes.append(ReasoningNode(node, clipped, left, min(right, start), False))
        else:
            removed.add(node)
    parents = {}
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("hyperedge must be an object")
        child = int(edge["target_step"])
        ps = tuple(sorted(set(map(int, edge.get("parent_steps", [])))))
        if child not in ids or any(p not in ids or p == child for p in ps):
            raise ValueError("invalid edge reference")
        if any(p in removed for p in ps):
            raise ValueError("removed answer-only node has outgoing dependency")
        if child not in removed:
            # Descendant closure uses every recorded parent-to-child incidence.
            parents[child] = tuple(sorted(set(ps) | set(parents.get(child, ()))))
    _validate_acyclic(tuple(n.node_id for n in nodes), parents)
    if not nodes:
        raise ValueError("graph_ineligible: no reasoning nodes after split")
    return Example(
        hashlib.sha256(problem.strip().encode()).hexdigest(),
        int(row.get("row_index", index)),
        problem.strip(),
        source,
        source[:start],
        block,
        str(gold).strip(),
        (start, end),
        tuple(nodes),
        parents,
        row,
    )


def load_examples(path, limit=0):
    examples, failures = [], []
    with open(path, encoding="utf-8") as reader:
        for index, line in enumerate(reader):
            if limit and index >= limit:
                break
            try:
                examples.append(parse_example(json.loads(line), index))
            except (ValueError, TypeError, KeyError) as error:
                split_valid = False
                try:
                    row = json.loads(line)
                    graph = row["hypergraph"]
                    if isinstance(graph, str):
                        graph = json.loads(graph)
                    terminal_span(graph, graph.get("solution") or graph.get("reference_solution"))
                    split_valid = True
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
                failures.append(
                    {
                        "line": index + 1,
                        "answer_split_valid": split_valid,
                        "graph_eligible": False,
                        "reason": str(error),
                    }
                )
    if not examples:
        raise ValueError(f"No valid graph examples: {failures[:3]}")
    return examples, {
        "sha256": file_sha256(path),
        "parser": PARSER_VERSION,
        "eligible": len(examples),
        "excluded": len(failures),
        "exclusion_reasons": dict(Counter(f["reason"] for f in failures)),
        "invalid_rows": failures,
    }


def render_mask(example, dropped, include_answer):
    retained = [node.text for node in example.nodes if node.node_id not in dropped]
    reasoning = "\n\n".join(retained)
    if not include_answer:
        return reasoning
    answer = example.final_answer_block.strip()
    if not re.match(r"Final\s+answer\s*:", answer, re.I):
        answer = "Final answer:\n" + answer
    return "\n\n".join(part for part in (reasoning, answer) if part)
