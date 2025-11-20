from __future__ import annotations
from typing import Dict, Optional
from ..domain.grid import Grid

class GridRepoInMemory:
    def __init__(self):
        self._grids: Dict[str, Grid] = {}
    def save(self, g: Grid) -> Grid:
        self._grids[g.id]=g; return g
    def get(self, grid_id: str) -> Optional[Grid]:
        return self._grids.get(grid_id)
    def all(self): return list(self._grids.values())
