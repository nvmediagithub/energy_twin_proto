from __future__ import annotations
from typing import List, Tuple, Optional
import math
import numpy as np
from ..domain.models import Village, Bridge
from ..domain.geometry import Point

def _polyline_to_path(points: List[Point], scale: float) -> str:
    if not points:
        return ""
    cmds = [f"M {points[0].x * scale:.2f} {points[0].y * scale:.2f}"]
    for p in points[1:]:
        cmds.append(f"L {p.x * scale:.2f} {p.y * scale:.2f}")
    return " ".join(cmds)

def _river_polys(points: List[Point], half_w: float) -> Tuple[List[Point], List[Point]]:
    left: List[Point] = []
    right: List[Point] = []
    for i, p in enumerate(points):
        if i == 0:
            q = points[i + 1]
        elif i == len(points) - 1:
            q = points[i - 1]
        else:
            q = points[i + 1]
        dx = q.x - p.x
        dy = q.y - p.y
        ln = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / ln, dx / ln
        left.append(Point(p.x + nx * half_w, p.y + ny * half_w))
        right.append(Point(p.x - nx * half_w, p.y - ny * half_w))
    return left, right

def _contours_to_paths(hmap: np.ndarray, levels: List[float]) -> List[List[Point]]:
    # very lightweight contour approximation: trace cells that cross level and emit short segments
    h, w = hmap.shape
    paths: List[List[Point]] = []
    for lvl in levels:
        segs=[]
        for y in range(h-1):
            for x in range(w-1):
                v00=hmap[y,x]; v10=hmap[y,x+1]; v01=hmap[y+1,x]; v11=hmap[y+1,x+1]
                mn=min(v00,v10,v01,v11); mx=max(v00,v10,v01,v11)
                if mn<=lvl<=mx:
                    # add a tiny polyline centered in cell for visual texture
                    segs.append([Point(x+0.2,y+0.5), Point(x+0.8,y+0.5)])
        # downsample segments
        if segs:
            paths.extend(segs[::6])
    return paths

def _bridge_rect(br: Bridge, scale: float) -> List[Tuple[float,float]]:
    # oriented rectangle corners
    L=br.length*scale; W=br.width*scale
    c=math.cos(br.rotation); s=math.sin(br.rotation)
    corners=[(-L/2,-W/2),(L/2,-W/2),(L/2,W/2),(-L/2,W/2)]
    pts=[]
    for x,y in corners:
        rx=x*c - y*s; ry=x*s + y*c
        pts.append((br.center.x*scale+rx, br.center.y*scale+ry))
    return pts

def village_to_svg(v: Village) -> str:
    scale_x = v.width / max(1.0, v.debug.get("grid_w", v.width))
    scale_y = v.height / max(1.0, v.debug.get("grid_h", v.height))
    scale = (scale_x + scale_y) / 2

    def sx(x): return x * scale
    def sy(y): return y * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{v.width}" height="{v.height}" viewBox="0 0 {v.width} {v.height}">',
        '<defs>'
        '  <pattern id="fieldHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(18)">'
        '    <line x1="0" y1="0" x2="0" y2="6" stroke="#7a7269" stroke-width="0.7" opacity="0.35"/>'
        '  </pattern>'
        '  <pattern id="bridgeHatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(90)">'
        '    <line x1="0" y1="0" x2="0" y2="5" stroke="#2d2a27" stroke-width="0.9" opacity="0.6"/>'
        '  </pattern>'
        '</defs>',
        '<rect width="100%" height="100%" fill="#b8c9ad"/>'
    ]

    # contour lines (background)
    if "heightmap" in v.debug:
        hmap = v.debug["heightmap"]
        levels = v.debug.get("heightmap_levels", [0.3,0.5,0.7])
        for seg in _contours_to_paths(hmap, levels):
            path = _polyline_to_path(seg, scale)
            parts.append(f'<path d="{path}" fill="none" stroke="#7e8c83" stroke-width="{0.6}" opacity="0.25" stroke-dasharray="2 4"/>')

    # river
    river = v.debug.get("river")
    if river:
        pts = [Point(p["x"], p["y"]) for p in river["points"]]
        half_w = river["width"] * 0.5
        l, r = _river_polys(pts, half_w)
        poly1 = l + r[::-1]
        pts_str1 = " ".join(f"{sx(p.x):.2f},{sy(p.y):.2f}" for p in poly1)
        parts.append(f'<polygon points="{pts_str1}" fill="#87b7c8" stroke="none" opacity="0.95"/>')

        l2, r2 = _river_polys(pts, half_w * 0.6)
        poly2 = l2 + r2[::-1]
        pts_str2 = " ".join(f"{sx(p.x):.2f},{sy(p.y):.2f}" for p in poly2)
        parts.append(f'<polygon points="{pts_str2}" fill="#6aa0b6" stroke="none" opacity="0.9"/>')

        shore1 = _polyline_to_path(l, scale)
        shore2 = _polyline_to_path(r, scale)
        parts.append(f'<path d="{shore1}" fill="none" stroke="#e7e0d1" stroke-width="{1.8 * scale}" opacity="0.9"/>')
        parts.append(f'<path d="{shore2}" fill="none" stroke="#e7e0d1" stroke-width="{1.8 * scale}" opacity="0.9"/>')

    # fields
    for fld in v.fields:
        pts = " ".join(f"{sx(p.x):.2f},{sy(p.y):.2f}" for p in fld.polygon)
        base_fill = "#b9b0a6" if fld.kind == "field" else "#a8b59d"
        parts.append(f'<polygon points="{pts}" fill="{base_fill}" stroke="#7a7269" stroke-width="0.7" opacity="0.95"/>')
        parts.append(f'<polygon points="{pts}" fill="url(#fieldHatch)" stroke="none" opacity="0.6"/>')

    # roads
    for r in v.roads:
        sw = 2.6 if r.kind == "major" else 1.6
        path = _polyline_to_path(r.polyline.points, scale)
        parts.append(f'<path d="{path}" fill="none" stroke="#e6decf" stroke-width="{sw * 1.9}" stroke-linecap="round" stroke-linejoin="round"/>')
        parts.append(f'<path d="{path}" fill="none" stroke="#6f675f" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')

    # bridges + docks
    for br in v.bridges:
        pts = _bridge_rect(br, scale)
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x,y in pts)
        parts.append(f'<polygon points="{pts_str}" fill="#c59a7a" stroke="#2d2a27" stroke-width="0.9"/>')
        parts.append(f'<polygon points="{pts_str}" fill="url(#bridgeHatch)" stroke="none" opacity="0.7"/>')

    # houses
    for h in v.houses:
        w2, h2 = h.width * scale, h.height * scale
        cx, cy = sx(h.center.x), sy(h.center.y)
        ang = h.rotation
        c = math.cos(ang); s = math.sin(ang)
        corners = [(-w2 / 2, -h2 / 2), (w2 / 2, -h2 / 2), (w2 / 2, h2 / 2), (-w2 / 2, h2 / 2)]
        pts=[]
        for x,y in corners:
            rx = x*c - y*s; ry = x*s + y*c
            pts.append(f"{cx+rx:.2f},{cy+ry:.2f}")
        pts_str = " ".join(pts)
        landmark = h.meta.get("landmark")
        if landmark:
            fill = "#a37b5c"  # darker landmark
        else:
            fill = "#c59a7a"
        parts.append(f'<polygon points="{pts_str}" fill="{fill}" stroke="#2d2a27" stroke-width="0.9"/>')

    # trees
    for t in v.trees:
        parts.append(f'<circle cx="{sx(t.center.x):.2f}" cy="{sy(t.center.y):.2f}" r="{t.radius * scale:.2f}" fill="#7a8f86" stroke="#1f2422" stroke-width="0.5"/>')

    parts.append("</svg>")
    return "\n".join(parts)
