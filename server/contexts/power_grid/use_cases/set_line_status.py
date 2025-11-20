from __future__ import annotations
from dataclasses import dataclass
from ..domain.grid import Grid

@dataclass
class SetLineStatus:
    def execute(self, g: Grid, line_id: str, status: str) -> Grid:
        if line_id not in g.lines:
            return g
        if status not in ("online","open","faulted"):
            return g
        g.lines[line_id].status = status
        return g
