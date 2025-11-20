from __future__ import annotations
from typing import List
import numpy as np
import math
from .rng import RNG
from ..domain.geometry import Point
from ..domain.models import House, FieldPlot

def _irregular_quad(cx, cy, w, h, rng: RNG) -> List[Point]:
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]
    out = []
    for x, y in pts:
        out.append(Point(x + rng.gauss(0, w * 0.08), y + rng.gauss(0, h * 0.08)))
    return out

def generate_fields(houses: List[House], rng: RNG, map_w: int, map_h: int) -> List[FieldPlot]:
    fields: List[FieldPlot] = []
    if not houses:
        return fields

    centers = np.array([[h.center.x, h.center.y] for h in houses], dtype=np.float32)
    cx, cy = centers[:, 0].mean(), centers[:, 1].mean()
    dists = np.sqrt((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2)
    order = np.argsort(dists)[::-1]  # outskirts first

    seed_idxs = order[: max(4, len(houses) // 8)]
    for idx in seed_idxs:
        hs = houses[int(idx)]
        fw = rng.uniform(20, 55)
        fh = rng.uniform(18, 48)
        ox = hs.center.x + rng.gauss(0, 6)
        oy = hs.center.y + rng.gauss(0, 6)
        ox = float(np.clip(ox, 5, map_w - 5))
        oy = float(np.clip(oy, 5, map_h - 5))
        poly = _irregular_quad(ox, oy, fw, fh, rng)
        kind = "orchard" if rng.rand() < 0.18 else "field"
        fields.append(FieldPlot(poly, kind=kind))

    for hs in rng._r.sample(houses, k=min(6, len(houses))):
        if rng.rand() < 0.5:
            fw = rng.uniform(7, 12)
            fh = rng.uniform(6, 10)
            poly = _irregular_quad(
                hs.center.x + rng.gauss(0, 3),
                hs.center.y + rng.gauss(0, 3),
                fw, fh, rng
            )
            fields.append(FieldPlot(poly, kind="field"))

    return fields
