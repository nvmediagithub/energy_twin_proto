from __future__ import annotations
import numpy as np
from .rng import RNG
from .noise import fractal_noise

def generate_heightmap(width: int, height: int, rng: RNG) -> np.ndarray:
    # independent noise stream by advancing RNG a bit
    _ = [rng.rand() for _ in range(64)]
    hmap = fractal_noise(width, height, rng, octaves=5)
    # normalize
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-9)
    return hmap
