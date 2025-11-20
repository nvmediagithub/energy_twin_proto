from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PointDTO(BaseModel):
    x: float
    y: float

class RoadDTO(BaseModel):
    id: str
    kind: str
    polyline: List[PointDTO]

class HouseDTO(BaseModel):
    id: str
    x: float
    y: float
    base_kw: float

class GeoSnapshotDTO(BaseModel):
    id: str
    width: int
    height: int
    roads: List[RoadDTO]
    houses: List[HouseDTO]
    river: Optional[Dict[str, Any]] = None
