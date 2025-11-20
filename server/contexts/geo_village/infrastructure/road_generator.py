from __future__ import annotations
from typing import List, Tuple
import numpy as np
import math
from .rng import RNG
from ..domain.geometry import Point, Polyline
from ..domain.models import Road

def _pick_density_peaks(density: np.ndarray, rng: RNG, k: int) -> List[Point]:
    h, w = density.shape
    flat = density.flatten()
    # take top candidates then sample
    cand = np.argsort(flat)[-k * 25:]
    cand = list(map(int, cand))
    cand = rng._r.sample(cand, k=min(len(cand), k))
    pts = [Point((i % w) + 0.5, (i // w) + 0.5) for i in cand]
    return pts

def _mst(points: List[Point]) -> List[Tuple[int, int]]:
    n = len(points)
    if n <= 1:
        return []
    in_tree = [False] * n
    in_tree[0] = True
    edges: List[Tuple[int, int]] = []
    dists = [float("inf")] * n
    parent = [-1] * n
    for i in range(1, n):
        dists[i] = points[0].dist(points[i])
        parent[i] = 0
    for _ in range(n - 1):
        j = min((i for i in range(n) if not in_tree[i]), key=lambda i: dists[i])
        in_tree[j] = True
        edges.append((parent[j], j))
        for i in range(n):
            if not in_tree[i]:
                d = points[j].dist(points[i])
                if d < dists[i]:
                    dists[i] = d
                    parent[i] = j
    return edges

def _noisy_path(a: Point, b: Point, rng: RNG, steps: int = 20, jitter: float = 4.0) -> Polyline:
    pts = [a]
    for s in range(1, steps):
        t = s / steps
        x = a.x * (1 - t) + b.x * t
        y = a.y * (1 - t) + b.y * t
        dx, dy = b.x - a.x, b.y - a.y
        ln = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / ln, dx / ln
        amp = math.sin(t * math.pi)  # 0 at ends, max in middle
        x += nx * rng.gauss(0, jitter) * amp
        y += ny * rng.gauss(0, jitter) * amp
        pts.append(Point(x, y))
    pts.append(b)
    return Polyline(pts).chaikin_smooth(2)

def generate_roads(
    density: np.ndarray,
    rng: RNG,
    major_count: int,
    minor_per_major: int,
    highway: bool = False,
) -> List[Road]:
    h, w = density.shape
    nodes = _pick_density_peaks(density, rng, k=major_count + 2)

    if highway:
        entries = [
            Point(2, rng.uniform(h * 0.2, h * 0.8)),
            Point(w - 3, rng.uniform(h * 0.2, h * 0.8)),
        ]
        nodes = entries + nodes

    edges = _mst(nodes)
    roads: List[Road] = []

    # majors from MST
    major_edges = edges[: major_count + (2 if highway else 0)]
    for i, j in major_edges:
        poly = _noisy_path(nodes[i], nodes[j], rng, steps=24, jitter=4.5)
        roads.append(Road(poly, kind="major"))

    # minors as short branches from nodes
    for npt in nodes:
        for _ in range(minor_per_major):
            ang = rng.uniform(0, math.tau)
            length = rng.uniform(18, 42)
            pts = [npt]
            cur = npt
            dirv = Point(math.cos(ang), math.sin(ang))
            for _s in range(20):
                step = length / 20
                jitter = Point(rng.gauss(0, 0.6), rng.gauss(0, 0.6))
                dirv = (dirv * 0.85 + jitter * 0.15).normalized()
                cur = Point(cur.x + dirv.x * step, cur.y + dirv.y * step)
                if cur.x < 3 or cur.y < 3 or cur.x > w - 4 or cur.y > h - 4:
                    break
                if density[int(cur.y), int(cur.x)] < 0.12:
                    break
                pts.append(cur)
            if len(pts) > 3:
                roads.append(Road(Polyline(pts).chaikin_smooth(1), kind="minor"))

    return roads
