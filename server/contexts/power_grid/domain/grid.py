from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from .entities import BaseNode, Source, Pole, Consumer, Line

@dataclass
class Grid:
    id: str
    nodes: Dict[str, BaseNode] = field(default_factory=dict)
    lines: Dict[str, Line] = field(default_factory=dict)
    meta: Dict[str, any] = field(default_factory=dict)

    def add_node(self, node: BaseNode):
        self.nodes[node.id] = node

    def add_line(self, line: Line):
        self.lines[line.id] = line

    def neighbors(self, node_id: str) -> List[str]:
        out=[]
        for ln in self.lines.values():
            if ln.status!="online": continue
            if ln.from_id==node_id: out.append(ln.to_id)
            if ln.to_id==node_id: out.append(ln.from_id)
        return out

    def sources(self) -> List[Source]:
        return [n for n in self.nodes.values() if isinstance(n, Source)]

    def consumers(self) -> List[Consumer]:
        return [n for n in self.nodes.values() if isinstance(n, Consumer)]
