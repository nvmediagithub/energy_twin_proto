from __future__ import annotations
import asyncio, json
from typing import Callable, Awaitable, Optional, List
from persistence.inmem_store import STORE
from contexts.power_grid.use_cases.simulate_tick import SimulateTick
from contexts.power_grid.use_cases.overload_zones import ComputeOverloadZones

class SimScheduler:
    def __init__(self, on_snapshot: Callable[[dict], Awaitable[None]]):
        self.on_snapshot = on_snapshot
        self._task: Optional[asyncio.Task] = None
        self._tick_uc = SimulateTick()
        self._zones_uc = ComputeOverloadZones()

    async def start(self, grid_id: str, dt: float):
        STORE.sim_running=True; STORE.sim_dt=dt; STORE.current_grid_id=grid_id
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def pause(self):
        STORE.sim_running=False

    async def step(self, n: int = 1):
        for _ in range(n):
            await self._one_tick()

    async def _loop(self):
        while True:
            if STORE.sim_running and STORE.current_grid_id:
                await self._one_tick()
                await asyncio.sleep(STORE.sim_dt)
            else:
                await asyncio.sleep(0.1)

    async def _one_tick(self):
        grid = STORE.grids.get(STORE.current_grid_id)
        if grid is None: return
        STORE.sim_t += STORE.sim_dt
        self._tick_uc.execute(grid, STORE.sim_t)
        zones = self._zones_uc.execute(grid)
        STORE.grids.save(grid)
        await self.on_snapshot(self._snapshot(grid, zones))

    def _snapshot(self, grid, zones):
        nodes=[{"id":n.id,"type":n.__class__.__name__.lower(),"x":n.position.x,"y":n.position.y,
                "props":dict(n.props),"state":dict(n.state),"status":n.status}
               for n in grid.nodes.values()]
        edges=[{"id":e.id,"from_id":e.from_id,"to_id":e.to_id,"type":"line",
                "props":dict(e.props),"state":dict(e.state),"status":e.status}
               for e in grid.lines.values()]
        return {"id":grid.id,"nodes":nodes,"edges":edges,
                "overlays":{"pole_load_zones":zones},
                "sim_time":float(grid.meta.get("sim_time",0.0)),
                "grid_version": STORE.grid_versions.get(grid.id,0)}
