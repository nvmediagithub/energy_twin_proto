from __future__ import annotations
from pydantic import BaseModel
class SimStartDTO(BaseModel):
    grid_id: str
    dt: float = 1.0
