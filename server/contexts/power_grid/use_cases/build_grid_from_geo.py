from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ..domain.grid import Grid
from ..domain.geometry import Point
from ..domain.entities import Source, Pole, Consumer, Line
from ..infrastructure.idgen import new_id

@dataclass
class BuildPolicy:
    pole_step: float = 10.0
    connect_radius: float = 12.0
    pole_capacity_kw: float = 60.0
    line_capacity_kw: float = 200.0

class BuildGridFromGeo:
    def __init__(self, policy: BuildPolicy = BuildPolicy()):
        self.policy = policy

    def execute(self, geo_snapshot: Dict[str, Any], grid_id: str) -> Grid:
        g = Grid(id=grid_id)

        w = geo_snapshot["width"]; h = geo_snapshot["height"]
        src = Source(
            id=new_id("src"),
            position=Point(w * 0.5, h * 0.5),
            props={"p_max_kw": 500.0, "voltage_out_kv": 10.0},
            state={"p_out_kw": 0.0, "q_out_kvar": 0.0, "v_kv": 10.0},
        )
        g.add_node(src)

        poles: List[Pole] = []
        for rd in geo_snapshot.get("roads", []):
            pts = rd["polyline"]
            if len(pts) < 2: continue
            acc = 0.0
            prev = Point(pts[0]["x"], pts[0]["y"])
            for i in range(1, len(pts)):
                cur = Point(pts[i]["x"], pts[i]["y"])
                seg_len = prev.dist(cur)
                if seg_len < 1e-6:
                    prev = cur; continue
                t = self.policy.pole_step - acc
                while t <= seg_len:
                    u = t / seg_len
                    p = Point(prev.x + (cur.x-prev.x)*u, prev.y + (cur.y-prev.y)*u)
                    pole = Pole(
                        id=new_id("pole"),
                        position=p,
                        props={
                            "capacity_kw": self.policy.pole_capacity_kw,
                            "radius": self.policy.connect_radius,
                            "voltage_kv": 0.4,
                            "class": "lv",
                        },
                        state={
                            "p_load_kw": 0.0,
                            "q_load_kvar": 0.0,
                            "s_load_kva": 0.0,
                            "v_kv": 0.0,
                            "overload_ratio": 0.0,
                            "island_id": 0,
                        },
                    )
                    g.add_node(pole); poles.append(pole)
                    t += self.policy.pole_step
                acc = max(0.0, self.policy.pole_step - (seg_len - (t-self.policy.pole_step)))
                prev = cur

        for p in poles:
            nearest=None; best=1e9
            for q in poles:
                if q.id==p.id: continue
                d=p.position.dist(q.position)
                if d<best: best=d; nearest=q
            if nearest and best<=self.policy.pole_step*1.6:
                g.add_line(Line(
                    id=new_id("line"),
                    from_id=p.id, to_id=nearest.id,
                    props={
                        "capacity_kw": self.policy.line_capacity_kw,
                        "capacity_kva": self.policy.line_capacity_kw * 1.05,
                        "length": best,
                        "r_pu": 0.0008,
                        "x_pu": 0.0012,
                        "voltage_kv": 10.0,
                    },
                    state={
                        "p_flow_kw": 0.0,
                        "q_flow_kvar": 0.0,
                        "s_flow_kva": 0.0,
                        "overload_ratio": 0.0,
                        "island_id": 0,
                    },
                ))
        if poles:
            nearest=min(poles, key=lambda p: p.position.dist(src.position))
            g.add_line(Line(
                id=new_id("line"),
                from_id=src.id, to_id=nearest.id,
                props={
                    "capacity_kw": self.policy.line_capacity_kw,
                    "capacity_kva": self.policy.line_capacity_kw * 1.1,
                    "length": src.position.dist(nearest.position),
                    "r_pu": 0.0005,
                    "x_pu": 0.0008,
                },
                state={
                    "p_flow_kw": 0.0,
                    "q_flow_kvar": 0.0,
                    "s_flow_kva": 0.0,
                    "overload_ratio": 0.0,
                    "island_id": 0,
                },
            ))

        for hs in geo_snapshot.get("houses", []):
            base_kw=hs.get("base_kw", 1.5)
            c=Consumer(
                id=hs["id"],
                position=Point(hs["x"], hs["y"]),
                props={
                    "base_kw": base_kw,
                    "profile": hs.get("profile", "residential"),
                    "cos_phi": hs.get("cos_phi", 0.95),
                    "voltage_kv": 0.4,
                },
                state={
                    "p_kw": base_kw,
                    "q_kvar": 0.0,
                    "s_kva": base_kw,
                    "v_kv": 0.0,
                    "supplied": False,
                    "island_id": 0,
                },
            )
            g.add_node(c)

        for c in g.consumers():
            nearest=None; best=1e9
            for p in poles:
                d=c.position.dist(p.position)
                if d<best: best=d; nearest=p
            if nearest and best<=nearest.props.get("radius", self.policy.connect_radius):
                g.add_line(Line(
                    id=new_id("svc"),
                    from_id=nearest.id, to_id=c.id,
                    props={
                        "capacity_kw": 15.0,
                        "capacity_kva": 16.0,
                        "length": best,
                        "voltage_kv": 0.4,
                        "r_pu": 0.0015,
                        "x_pu": 0.0025,
                        "switchable": True,
                    },
                    state={
                        "p_flow_kw": 0.0,
                        "q_flow_kvar": 0.0,
                        "s_flow_kva": 0.0,
                        "overload_ratio": 0.0,
                        "island_id": 0,
                    },
                ))
        return g
