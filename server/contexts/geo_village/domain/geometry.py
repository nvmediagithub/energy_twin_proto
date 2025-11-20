from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Point":
        return Point(self.x * k, self.y * k)

    def dist(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Point":
        n = self.norm()
        return Point(self.x / n, self.y / n) if n > 1e-9 else Point(0.0, 0.0)

@dataclass
class Polyline:
    points: List[Point]

    def length(self) -> float:
        return sum(self.points[i].dist(self.points[i+1]) for i in range(len(self.points)-1))

    def chaikin_smooth(self, iters: int = 2) -> "Polyline":
        pts = self.points
        for _ in range(iters):
            new_pts = [pts[0]]
            for i in range(len(pts)-1):
                p = pts[i]; q = pts[i+1]
                new_pts.append(Point(0.75*p.x + 0.25*q.x, 0.75*p.y + 0.25*q.y))
                new_pts.append(Point(0.25*p.x + 0.75*q.x, 0.25*p.y + 0.75*q.y))
            new_pts.append(pts[-1])
            pts = new_pts
        return Polyline(pts)

    def sample_along(self, step: float) -> List[Tuple[Point, Point]]:
        # returns list of (position, direction) tuples
        out = []
        if len(self.points) < 2: return out
        seg_i = 0
        p0 = self.points[0]
        remain = step
        while seg_i < len(self.points)-1:
            p1 = self.points[seg_i+1]
            seg_len = p0.dist(p1)
            if seg_len < 1e-9:
                seg_i += 1; p0 = p1; continue
            if remain <= seg_len:
                t = remain / seg_len
                pos = Point(p0.x + (p1.x - p0.x)*t, p0.y + (p1.y - p0.y)*t)
                dir_vec = (p1 - p0).normalized()
                out.append((pos, dir_vec))
                p0 = pos
                remain = step
            else:
                remain -= seg_len
                seg_i += 1
                p0 = p1
        return out
