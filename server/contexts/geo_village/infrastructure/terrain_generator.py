from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math
from .rng import RNG
from ..domain.geometry import Point, Polyline

@dataclass
class River:
    centerline: Polyline
    width: float  # in grid units

def generate_river(width: int, height: int, rng: RNG, meander: float = 0.18) -> River:
    """Make a left->right meandering river using a smoothed random walk."""
    steps = 26
    margin = 10
    xs = np.linspace(margin, width - margin, steps)

    y = rng.uniform(height * 0.28, height * 0.72)
    ys: List[float] = []
    for i in range(steps):
        y += rng.gauss(0, height * meander / steps) * height * 0.35
        y = float(np.clip(y, margin, height - margin))
        ys.append(y)

    pts = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
    poly = Polyline(pts).chaikin_smooth(3)
    rwidth = rng.uniform(8.0, 14.0)
    return River(poly, width=rwidth)

def rasterize_river_distance(river: River, width: int, height: int) -> np.ndarray:
    """Brute distance-to-centerline, local window per point."""
    dist = np.full((height, width), np.inf, dtype=np.float32)
    pts = river.centerline.points
    for p in pts:
        ix, iy = int(p.x), int(p.y)
        x0 = max(0, ix - 28); x1 = min(width, ix + 29)
        y0 = max(0, iy - 28); y1 = min(height, iy + 29)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dloc = np.sqrt((xx - p.x) ** 2 + (yy - p.y) ** 2)
        dist[y0:y1, x0:x1] = np.minimum(dist[y0:y1, x0:x1], dloc)
    return dist
