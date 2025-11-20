from __future__ import annotations
from typing import List, Dict, Any
from ..domain.grid import Grid
from ..domain.entities import Pole

class ComputeOverloadZones:
    def execute(self, g: Grid) -> List[Dict[str, Any]]:
        zones=[]
        for n in g.nodes.values():
            if isinstance(n, Pole):
                ratio=float(n.state.get("overload_ratio",0.0))
                color="green" if ratio<0.6 else ("yellow" if ratio<0.9 else "red")
                zones.append({
                    "pole_id": n.id,
                    "x": n.position.x, "y": n.position.y,
                    "radius": float(n.props.get("radius",12.0)),
                    "overload_ratio": ratio,
                    "color": color,
                })
        return zones
