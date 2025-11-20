from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
from .geometry import Point

@dataclass
class BaseNode:
    id: str
    position: Point
    props: Dict[str, Any] = field(default_factory=dict)  # editable metadata
    state: Dict[str, Any] = field(default_factory=dict)  # computed simulation state
    status: str = "online"  # online/offline marker for UI

@dataclass
class Source(BaseNode):
    kind: str = "plant"  # HV source (plant/substation)
    # props: p_max_kw, voltage_out_kv
    # state: p_out_kw, q_out_kvar, v_kv

@dataclass
class Transformer(BaseNode):
    kind: str = "tp"
    # props: capacity_kva, voltage_in_kv, voltage_out_kv, efficiency
    # state: p_in_kw, q_in_kvar, p_out_kw, q_out_kvar, overload_ratio, v_kv

@dataclass
class Pole(BaseNode):
    kind: str = "pole"
    # props: capacity_kw, radius, voltage_kv, class ("hv"/"lv")
    # state: p_load_kw, q_load_kvar, s_load_kva, v_kv, overload_ratio, island_id

@dataclass
class Consumer(BaseNode):
    kind: str = "house"
    # props: base_kw, profile, cos_phi, voltage_kv
    # state: p_kw, q_kvar, s_kva, v_kv, supplied, island_id

@dataclass
class Line:
    id: str
    from_id: str
    to_id: str
    props: Dict[str, Any] = field(default_factory=dict)
    # capacity_kva (или capacity_kw), length, voltage_kv, r_pu, x_pu, switchable(bool)
    state: Dict[str, Any] = field(default_factory=dict)
    # p_flow_kw, q_flow_kvar, s_flow_kva, loss_kw, overload_ratio, island_id
    status: str = "online"  # online/open/faulted
