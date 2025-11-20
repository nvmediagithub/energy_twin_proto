from __future__ import annotations
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json
from persistence.inmem_store import STORE
from services.sim_scheduler import SimScheduler

router=APIRouter(prefix="/sim", tags=["sim"])
clients: List[WebSocket]=[]

async def broadcast(snapshot:dict):
    msg=json.dumps(snapshot); dead=[]
    for ws in clients:
        try: await ws.send_text(msg)
        except Exception: dead.append(ws)
    for ws in dead:
        clients.remove(ws)

scheduler=SimScheduler(on_snapshot=broadcast)

@router.post("/start")
async def start(grid_id:str, dt:float=1.0):
    await scheduler.start(grid_id, dt); return {"running":True}

@router.post("/pause")
async def pause():
    await scheduler.pause(); return {"running":False}

@router.post("/step")
async def step(n:int=1):
    await scheduler.step(n); return {"ok":True}

@router.get("/state")
def state():
    return {"running":STORE.sim_running,"t":STORE.sim_t,
            "grid_version":STORE.grid_versions.get(STORE.current_grid_id or "",0)}

@router.websocket("/stream")
async def stream(ws:WebSocket):
    await ws.accept(); clients.append(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        if ws in clients: clients.remove(ws)
