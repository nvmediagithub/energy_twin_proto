from __future__ import annotations
from pydantic import BaseModel
from typing import Dict, Any, List

class NodeDTO(BaseModel):
    id: str; type: str; x: float; y: float
    props: Dict[str, Any]; state: Dict[str, Any]; status: str

class EdgeDTO(BaseModel):
    id: str; from_id: str; to_id: str; type: str
    props: Dict[str, Any]; state: Dict[str, Any]; status: str

class GridSnapshotDTO(BaseModel):
    id: str
    nodes: List[NodeDTO]
    edges: List[EdgeDTO]
    overlays: Dict[str, Any]
    sim_time: float
    grid_version: int
