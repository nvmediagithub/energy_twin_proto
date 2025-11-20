from __future__ import annotations
import random
from dataclasses import dataclass

@dataclass
class RNG:
    seed: int

    def __post_init__(self):
        self._r = random.Random(self.seed)

    def rand(self) -> float:
        return self._r.random()

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def choice(self, xs):
        return self._r.choice(xs)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._r.gauss(mu, sigma)

    def shuffle(self, xs):
        self._r.shuffle(xs); return xs
