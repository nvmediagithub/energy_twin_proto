from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .geometry import Point, Polyline

@dataclass
class Road:
    polyline: Polyline
    kind: str = "minor"  # 'major' or 'minor'

@dataclass
class House:
    center: Point
    width: float
    height: float
    rotation: float  # radians
    side: str  # 'left' or 'right'
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FieldPlot:
    polygon: List[Point]
    kind: str = "field"  # field/orchard/square

@dataclass
class Tree:
    center: Point
    radius: float

@dataclass
class Bridge:
    center: Point
    length: float
    width: float
    rotation: float  # along the road

@dataclass
class Village:
    width: int
    height: int
    tags: List[str]
    roads: List[Road] = field(default_factory=list)
    houses: List[House] = field(default_factory=list)
    fields: List[FieldPlot] = field(default_factory=list)
    trees: List[Tree] = field(default_factory=list)
    bridges: List[Bridge] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)
