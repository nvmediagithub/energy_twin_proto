from __future__ import annotations
from typing import List, Tuple
import math
from .rng import RNG
from ..domain.geometry import Point
from ..domain.models import Road, House

def place_houses(roads: List[Road], rng: RNG, density_factor: float) -> List[House]:
    houses: List[House] = []
    for road in roads:
        step = 5.0 if road.kind=="major" else 7.0
        step /= density_factor
        for pos, dirv in road.polyline.sample_along(step=step):
            # determine normal
            nx, ny = -dirv.y, dirv.x
            side = "left" if rng.rand() < 0.5 else "right"
            sign = 1.0 if side=="left" else -1.0
            offset = rng.uniform(2.0, 4.5) / density_factor
            cx, cy = pos.x + nx*offset*sign, pos.y + ny*offset*sign
            width = rng.uniform(2.0, 4.0) / density_factor
            height = rng.uniform(2.0, 4.5) / density_factor
            rot = math.atan2(dirv.y, dirv.x) + (math.pi/2 if side=="left" else -math.pi/2) + rng.gauss(0, 0.15)

            # simple collision check
            cpt = Point(cx, cy)
            if any(cpt.dist(h.center) < (h.width+h.height)*0.75 for h in houses):
                continue

            houses.append(House(cpt, width, height, rot, side))
    return houses
