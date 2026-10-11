"""Linear Beta exposure policy and historical feedback standardization."""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta


class LinearBeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(2, 4))
        self.c = nn.Parameter(torch.full((2,), math.log(math.expm1(1))))

    def distribution(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).detach()
        a, b = 1 + torch.nn.functional.softplus(self.W @ state + self.c)
        return Beta(a, b)

    def sample(self, state):
        with torch.no_grad():
            return float(self.distribution(state).sample())

class Standardizer:
    def __init__(self, decay, floor, clip):
        self.decay, self.floor, self.clip = decay, floor, clip
        self.count, self.mean, self.variance = 0, 0.0, 0.0

    def z(self, value):
        return (
            float(
                np.clip(
                    (value - self.mean) / math.sqrt(max(self.variance, self.floor)), -self.clip, self.clip
                )
            )
            if self.count
            else 0.0
        )

    def ingest(self, value):
        if not self.count:
            self.mean = value
        else:
            old = self.mean
            self.mean = self.decay * old + (1 - self.decay) * value
            self.variance = self.decay * self.variance + (1 - self.decay) * (value - old) * (
                value - self.mean
            )
        self.count += 1

    def diagnostics(self):
        return {**vars(self), "scale": math.sqrt(max(self.variance, self.floor))}
