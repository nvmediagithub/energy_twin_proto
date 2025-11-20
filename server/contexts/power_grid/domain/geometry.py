from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    def dist(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
