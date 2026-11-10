"""Checkpointable shuffled question sampler."""

import numpy as np


class Sampler:
    def __init__(self, ids, seed):
        self.ids = list(ids)
        self.rng = np.random.default_rng(seed)
        self.order, self.cursor = [], 0

    def take(self, count):
        result = []
        while len(result) < count:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.ids).tolist()
                self.cursor = 0
            size = min(count - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor : self.cursor + size])
            self.cursor += size
        return result

    def state_dict(self):
        return {
            "ids": self.ids,
            "order": self.order,
            "cursor": self.cursor,
            "rng": self.rng.bit_generator.state,
        }

    def restore(self, state):
        if state["ids"] != self.ids:
            raise ValueError("Sampler IDs differ")
        self.order, self.cursor = state["order"], state["cursor"]
        self.rng.bit_generator.state = state["rng"]
