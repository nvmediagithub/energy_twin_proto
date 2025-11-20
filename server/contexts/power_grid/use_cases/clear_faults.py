from __future__ import annotations
from dataclasses import dataclass
from ..domain.grid import Grid

@dataclass
class ClearFaults:
    def execute(self, g: Grid) -> Grid:
        for l in g.lines.values():
            if l.status=="faulted":
                l.status="open"   # после аварии обычно остаётся разомкнутой
        return g
