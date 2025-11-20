from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from api.routes_geo import router as geo_router
from api.routes_grid import router as grid_router
from api.routes_sim import router as sim_router

app=FastAPI(title="Energy Twin Prototype")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(geo_router); app.include_router(grid_router); app.include_router(sim_router)
app.mount("/", StaticFiles(directory="../client", html=True), name="client")

if __name__=="__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
