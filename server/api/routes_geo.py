from __future__ import annotations
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import random, uuid
from persistence.inmem_store import STORE
from contexts.geo_village.use_cases.generate_village import GenerateVillage, GenerateVillageParams

router = APIRouter(prefix="/geo", tags=["geo"])

@router.post("/generate")
def generate_geo(seed:int=0,width:int=300,height:int=300,tags:str=""):
    tag_list=[t.strip() for t in tags.split(",") if t.strip()]
    v=GenerateVillage(GenerateVillageParams(seed=seed,width=width,height=height,tags=tag_list)).execute()
    geo_id=f"geo_{uuid.uuid4().hex[:8]}"
    houses=[]
    for i,h in enumerate(v.houses):
        base_kw=1.0+random.random()*2.0
        houses.append({"id":f"house_{i}","x":h.center.x,"y":h.center.y,"base_kw":base_kw})
    roads=[]
    for i,r in enumerate(v.roads):
        roads.append({"id":f"road_{i}","kind":r.kind,
                      "polyline":[{"x":p.x,"y":p.y} for p in r.polyline.points]})
    snap={"id":geo_id,"width":v.width,"height":v.height,
          "houses":houses,"roads":roads,"river":v.debug.get("river")}
    STORE.geos[geo_id]=snap
    return snap

@router.get("/{geo_id}")
def get_geo(geo_id:str):
    snap=STORE.geos.get(geo_id)
    if not snap: raise HTTPException(404,"geo not found")
    return snap
