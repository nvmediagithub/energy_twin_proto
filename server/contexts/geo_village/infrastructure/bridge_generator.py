from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import math
from .rng import RNG
from ..domain.geometry import Point
from ..domain.models import Road, Bridge

def _segment_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> Optional[Tuple[float,float,Point]]:
    # returns (t,u,point) where point = p1 + t*(p2-p1)
    def cross(a: Point, b: Point) -> float:
        return a.x*b.y - a.y*b.x
    r = p2 - p1
    s = q2 - q1
    denom = cross(r, s)
    if abs(denom) < 1e-9:
        return None
    qp = q1 - p1
    t = cross(qp, s) / denom
    u = cross(qp, r) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        pt = Point(p1.x + r.x*t, p1.y + r.y*t)
        return t, u, pt
    return None

def _polyline_segments(points):
    for i in range(len(points)-1):
        yield points[i], points[i+1]

def generate_bridges(roads: List[Road], river_pts: List[Point], rng: RNG) -> List[Bridge]:
    bridges: List[Bridge] = []
    if not river_pts:
        return bridges

    river_segs = list(_polyline_segments(river_pts))

    for road in roads:
        for a,b in _polyline_segments(road.polyline.points):
            for c,d in river_segs:
                inter = _segment_intersect(a,b,c,d)
                if inter is None:
                    continue
                _, _, pt = inter
                # avoid duplicates close together
                if any(pt.dist(br.center) < 6.0 for br in bridges):
                    continue
                # orientation along road
                ang = math.atan2(b.y-a.y, b.x-a.x)
                length = rng.uniform(9.0, 16.0) if road.kind=="major" else rng.uniform(7.0, 12.0)
                width = rng.uniform(2.0, 3.2)
                bridges.append(Bridge(center=pt, length=length, width=width, rotation=ang))
    return bridges

def generate_docks(bridges: List[Bridge], rng: RNG) -> List[Bridge]:
    docks: List[Bridge] = []
    for br in bridges:
        if rng.rand() < 0.55:
            # 1-2 small docks near bridge
            for _ in range(rng.randint(1,2)):
                ang = br.rotation + (math.pi/2 if rng.rand()<0.5 else -math.pi/2) + rng.gauss(0,0.2)
                length = rng.uniform(7.0, 12.0)
                width = rng.uniform(1.2, 1.8)
                offset = rng.uniform(3.0, 6.0)
                cx = br.center.x + math.cos(ang)*offset
                cy = br.center.y + math.sin(ang)*offset
                docks.append(Bridge(center=Point(cx,cy), length=length, width=width, rotation=ang))
    return docks
