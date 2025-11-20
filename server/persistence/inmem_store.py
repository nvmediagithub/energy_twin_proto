from __future__ import annotations
from typing import Dict, Any, Optional
from contexts.power_grid.infrastructure.repo_inmem import GridRepoInMemory

class InMemStore:
    def __init__(self):
        self.geos: Dict[str, Any] = {}
        self.grids = GridRepoInMemory()
        self.grid_versions: Dict[str, int] = {}
        self.sim_running: bool = False
        self.sim_t: float = 0.0
        self.sim_dt: float = 1.0
        self.current_grid_id: Optional[str] = None

STORE = InMemStore()
