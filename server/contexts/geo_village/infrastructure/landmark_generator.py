from __future__ import annotations
from typing import List
import numpy as np
import math
from .rng import RNG
from ..domain.geometry import Point
from ..domain.models import House, Road

def add_landmarks(houses: List[House], roads: List[Road], rng: RNG, count: int=2) -> List[House]:
    if not houses:
        return houses
    # center of village by houses
    centers = np.array([[h.center.x, h.center.y] for h in houses], dtype=np.float32)
    cx, cy = centers[:,0].mean(), centers[:,1].mean()
    # pick closest houses to center, replace with landmarks
    dists = np.sqrt((centers[:,0]-cx)**2 + (centers[:,1]-cy)**2)
    order = np.argsort(dists)
    for idx in order[:count]:
        h = houses[int(idx)]
        h.width *= rng.uniform(2.0, 3.0)
        h.height *= rng.uniform(1.8, 2.8)
        h.meta["landmark"] = rng.choice(["hall","barn","chapel"])
        h.meta["color_variant"] = "dark"
    return houses
