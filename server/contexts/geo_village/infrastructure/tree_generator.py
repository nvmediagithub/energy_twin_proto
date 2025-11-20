from __future__ import annotations
from typing import List
import numpy as np
import math
from .rng import RNG
from ..domain.geometry import Point
from ..domain.models import Tree, Road, House

def scatter_trees(
    density: np.ndarray,
    roads: List[Road],
    houses: List[House],
    rng: RNG,
    count: int,
    forest_blobs: int = 3,
) -> List[Tree]:
    h, w = density.shape
    trees: List[Tree] = []
    road_pts = [p for r in roads for p in r.polyline.points]

    def near_road(pt: Point, d=3.0) -> bool:
        return any(pt.dist(rp) < d for rp in road_pts)

    def near_house(pt: Point, d=4.0) -> bool:
        return any(pt.dist(hs.center) < d for hs in houses)

    # forest blobs in outer low-density areas
    for _ in range(forest_blobs):
        cx = rng.uniform(w * 0.1, w * 0.9)
        cy = rng.uniform(h * 0.1, h * 0.9)
        if density[int(cy), int(cx)] > 0.25:
            continue
        rx = rng.uniform(18, 35)
        ry = rng.uniform(18, 35)
        ntrees = rng.randint(80, 150)
        for _t in range(ntrees):
            ang = rng.uniform(0, math.tau)
            r = abs(rng.gauss(0, 0.45))
            x = cx + math.cos(ang) * rx * r + rng.gauss(0, 1.0)
            y = cy + math.sin(ang) * ry * r + rng.gauss(0, 1.0)
            if x < 2 or y < 2 or x > w - 3 or y > h - 3:
                continue
            pt = Point(x, y)
            if near_road(pt, d=2.3) or near_house(pt, d=5.0):
                continue
            trees.append(Tree(pt, radius=rng.uniform(1.1, 2.2)))

    # orchards: small grids near some houses
    for hs in houses:
        if rng.rand() < 0.08:
            base = hs.center
            cols = rng.randint(3, 5)
            rows = rng.randint(3, 5)
            spacing = rng.uniform(2.0, 2.6)
            for i in range(cols):
                for j in range(rows):
                    x = base.x + (i - (cols - 1) / 2) * spacing + rng.gauss(0, 0.2)
                    y = base.y + (j - (rows - 1) / 2) * spacing + rng.gauss(0, 0.2)
                    pt = Point(x, y)
                    if x < 1 or y < 1 or x > w - 2 or y > h - 2:
                        continue
                    if near_house(pt, d=2.2):
                        continue
                    trees.append(Tree(pt, radius=rng.uniform(0.8, 1.3)))

    # remaining scattered trees
    tries = 0
    while len(trees) < count and tries < count * 50:
        tries += 1
        x = rng.uniform(0, w - 1)
        y = rng.uniform(0, h - 1)
        dval = density[int(y), int(x)]
        if dval > 0.5:
            continue
        pt = Point(x + 0.5, y + 0.5)
        if near_house(pt) or near_road(pt):
            continue
        if any(pt.dist(t.center) < 2.6 for t in trees):
            continue
        trees.append(Tree(pt, radius=rng.uniform(0.7, 1.5)))

    return trees
