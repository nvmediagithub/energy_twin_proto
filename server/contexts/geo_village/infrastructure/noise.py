from __future__ import annotations
import numpy as np
from .rng import RNG

def value_noise(width: int, height: int, rng: RNG, grid: int = 8) -> np.ndarray:
    # random grid then bilinear upsample
    gx = width // grid + 2
    gy = height // grid + 2
    base = np.array([[rng.rand() for _ in range(gx)] for __ in range(gy)], dtype=np.float32)
    # coordinates in base grid space
    ys = np.linspace(0, gy-1, height)
    xs = np.linspace(0, gx-1, width)
    xi = np.floor(xs).astype(int); xf = xs - xi
    yi = np.floor(ys).astype(int); yf = ys - yi
    out = np.zeros((height, width), dtype=np.float32)
    for j in range(height):
        y0 = yi[j]; y1 = min(y0+1, gy-1); ty = yf[j]
        row0 = base[y0]; row1 = base[y1]
        v0 = row0[xi]*(1-xf) + row0[np.minimum(xi+1, gx-1)]*xf
        v1 = row1[xi]*(1-xf) + row1[np.minimum(xi+1, gx-1)]*xf
        out[j] = v0*(1-ty) + v1*ty
    return out

def fractal_noise(width: int, height: int, rng: RNG, octaves: int = 4) -> np.ndarray:
    acc = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        grid = max(2, 2**(o+2))
        n = value_noise(width, height, rng, grid=grid)
        acc += n * amp
        total += amp
        amp *= 0.5
    return acc / total
