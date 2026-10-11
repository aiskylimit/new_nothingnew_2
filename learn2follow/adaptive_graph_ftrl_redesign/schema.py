"""Verified reasoning spans, balanced boxed answers and DAG validation."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReasoningNode:
    node_id: int
    text: str
    char_start: int | None = None
    char_end: int | None = None
    protected: bool = False

class SchemaError(ValueError):
    """Raised when a row does not match the source hypergraph schema."""

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _boxed_span(text: str) -> tuple[int, int, int] | None:
    """Return the command start, opening brace and exclusive closing end."""
    if not isinstance(text, str):
        return None
    starts = [match.start() for match in re.finditer(r"\\boxed(?![A-Za-z])", text)]
    for start in starts[-1:]:
        brace = start + len("\\boxed")
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace == len(text) or text[brace] != "{":
            return None
        depth = 0
        for pos in range(brace, len(text)):
            char = text[pos]
            backslashes = 0
            previous = pos - 1
            while previous >= 0 and text[previous] == "\\":
                backslashes += 1
                previous -= 1
            if backslashes % 2:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return start, brace, pos + 1
    return None

def _validate_acyclic(node_ids: tuple[int, ...], parents_by_child: dict[int, tuple[int, ...]]) -> None:
    temporary: set[int] = set()
    permanent: set[int] = set()

    def visit(node: int):
        if node in permanent:
            return
        if node in temporary:
            raise SchemaError("dependency graph contains a cycle")
        temporary.add(node)
        for parent in parents_by_child.get(node, ()):
            visit(parent)
        temporary.remove(node)
        permanent.add(node)

    for node in node_ids:
        visit(node)
