from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import uuid
from persistence.inmem_store import STORE
from contexts.power_grid.use_cases.build_grid_from_geo import BuildGridFromGeo
from contexts.power_grid.use_cases.set_line_status import SetLineStatus
from contexts.power_grid.use_cases.trigger_fault import TriggerFault
from contexts.power_grid.use_cases.clear_faults import ClearFaults
from contexts.power_grid.infrastructure.idgen import new_id
from contexts.power_grid.domain.entities import Source, Pole, Line
from contexts.power_grid.domain.geometry import Point

router=APIRouter(prefix="/grid", tags=["grid"])
builder=BuildGridFromGeo()
SET_LINE_STATUS=SetLineStatus()
TRIGGER_FAULT=TriggerFault()
CLEAR_FAULTS=ClearFaults()

@router.post("/build_from_geo")
def build_from_geo(geo_id:str):
    geo=STORE.geos.get(geo_id)
    if not geo: raise HTTPException(404,"geo not found")
    grid_id=f"grid_{uuid.uuid4().hex[:8]}"
    g=builder.execute(geo, grid_id)
    STORE.grids.save(g); STORE.grid_versions[grid_id]=0
    return snapshot(g)

@router.get("/{grid_id}")
def get_grid(grid_id:str):
    g=STORE.grids.get(grid_id)
    if not g: raise HTTPException(404,"grid not found")
    return snapshot(g)

@router.patch("/{grid_id}/command")
def command(grid_id:str, cmd:Dict[str,Any]):
    g=STORE.grids.get(grid_id)
    if not g: raise HTTPException(404,"grid not found")
    ctype=cmd.get("type"); payload=cmd.get("payload",{})
    if ctype=="update_props":
        nid=payload["node_id"]; props=payload.get("props",{})
        node=g.nodes.get(nid)
        if node: node.props.update(props)
    elif ctype=="add_source":
        pos=payload["position"]
        g.add_node(Source(id=new_id("src"), position=Point(pos["x"],pos["y"]),
                          props={"p_max_kw":payload.get("p_max_kw",500.0)}))
    elif ctype=="add_pole":
        pos=payload["position"]
        g.add_node(Pole(id=new_id("pole"), position=Point(pos["x"],pos["y"]),
                        props={"capacity_kw":payload.get("capacity_kw",60.0),
                               "radius":payload.get("radius",12.0)}))
    elif ctype=="connect_line":
        a=payload["from_id"]; b=payload["to_id"]
        if a in g.nodes and b in g.nodes:
            g.add_line(Line(id=new_id("line"), from_id=a, to_id=b,
                            props={"capacity_kw":payload.get("capacity_kw",200.0),
                                   "length": g.nodes[a].position.dist(g.nodes[b].position)}))
    elif ctype=="set_line_status":
        g = SET_LINE_STATUS.execute(g, payload["line_id"], payload["status"])
    elif ctype=="trigger_fault":
        g = TRIGGER_FAULT.execute(g, payload.get("line_id"))
    elif ctype=="clear_faults":
        g = CLEAR_FAULTS.execute(g)
    else:
        raise HTTPException(400,f"unknown command {ctype}")
    STORE.grids.save(g); STORE.grid_versions[grid_id]=STORE.grid_versions.get(grid_id,0)+1
    return snapshot(g)

def snapshot(g):
    nodes=[{"id":n.id,"type":n.__class__.__name__.lower(),"x":n.position.x,"y":n.position.y,
            "props":dict(n.props),"state":dict(n.state),"status":n.status}
           for n in g.nodes.values()]
    edges=[{"id":e.id,"from_id":e.from_id,"to_id":e.to_id,"type":"line",
            "props":dict(e.props),"state":dict(e.state),"status":e.status}
           for e in g.lines.values()]
    return {"id":g.id,"nodes":nodes,"edges":edges,
            "overlays":{"pole_load_zones":[]},
            "sim_time":float(g.meta.get("sim_time",0.0)),
            "grid_version": STORE.grid_versions.get(g.id,0)}
