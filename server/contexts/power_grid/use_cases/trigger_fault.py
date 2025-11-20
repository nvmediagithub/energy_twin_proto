from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import random
from ..domain.grid import Grid

@dataclass
class TriggerFault:
    probability: float = 1.0  # 1.0 = всегда

    def execute(self, g: Grid, line_id: Optional[str] = None) -> Grid:
        if random.random() > self.probability:
            return g

        if line_id and line_id in g.lines:
            g.lines[line_id].status="faulted"
            return g

        # random online line
        candidates=[l for l in g.lines.values() if l.status=="online"]
        if candidates:
            random.choice(candidates).status="faulted"
        return g
