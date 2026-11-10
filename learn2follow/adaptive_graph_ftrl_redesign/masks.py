"""Graph-only candidate pool; sample unique retained-token exposure first."""

from dataclasses import dataclass
import hashlib
from itertools import combinations

import numpy as np

def children_by_parent(example: object) -> dict[int, set[int]]:
    children: dict[int, set[int]] = {node.node_id: set() for node in example.nodes}
    for child, parents in example.parents_by_child.items():
        for parent in parents:
            children.setdefault(parent, set()).add(child)
    return children

def descendant_closure(example: object, seeds: set[int] | frozenset[int]) -> frozenset[int]:
    valid = set(example.node_ids)
    dropped = set(seeds)
    if not dropped <= valid:
        raise ValueError(f"unknown node ids in seed: {sorted(dropped - valid)}")
    children = children_by_parent(example)
    changed = True
    while changed:
        changed = False
        for node in list(dropped):
            for child in children.get(node, ()):
                if child not in dropped:
                    dropped.add(child)
                    changed = True
    return frozenset(dropped)


@dataclass
class Pool:
    groups: dict
    total_tokens: int
    token_lengths: dict
    pair_total: int
    pair_used: int

    def sample(self, alpha, tau, rng):
        levels = sorted(self.groups)
        scores = -np.abs(np.asarray(levels) / self.total_tokens - alpha) / tau
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        count = levels[int(rng.choice(len(levels), p=probs))]
        masks = self.groups[count]
        mask = masks[int(rng.integers(len(masks)))]
        return mask, count / self.total_tokens


def candidate_pool(example, tokenizer, variant="graph", pair_limit=4096, seed=42):
    lengths = {
        node.node_id: len(tokenizer.encode(node.text, add_special_tokens=False)) for node in example.nodes
    }
    total = sum(lengths.values())
    if total <= 0:
        raise ValueError("graph_ineligible: zero reasoning tokens")
    ids = example.node_ids
    pair_total = len(ids) * (len(ids) - 1) // 2
    pair_used = 0
    if variant == "prefix":
        masks = {frozenset(ids[n:]) for n in range(len(ids))}
    else:
        closures = [descendant_closure(example, {node}) for node in ids]
        masks = set(closures)
        # Reservoir sampling bounds pair storage while preserving a stable seed.
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(f"{seed}:{example.qid}:pairs".encode()).digest()[:8], "big")
        )
        pairs = []
        for seen, pair in enumerate(combinations(range(len(ids)), 2)):
            if pair_limit == 0 or len(pairs) < pair_limit:
                pairs.append(pair)
            else:
                slot = int(rng.integers(seen + 1))
                if slot < pair_limit:
                    pairs[slot] = pair
        pair_used = len(pairs)
        masks.update(closures[a] | closures[b] for a, b in pairs)
    masks.add(frozenset(ids))
    masks.discard(frozenset())
    groups = {}
    for mask in sorted(masks, key=lambda m: tuple(sorted(m))):
        count = sum(lengths[v] for v in ids if v not in mask)
        groups.setdefault(count, []).append(mask)
    return Pool(groups, total, lengths, pair_total if variant == "graph" else 0, pair_used)
