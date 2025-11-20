from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from ..domain.models import Village
from ..infrastructure.rng import RNG
from ..infrastructure.noise import fractal_noise
from ..infrastructure.road_generator import generate_roads
from ..infrastructure.house_placer import place_houses
from ..infrastructure.field_generator import generate_fields
from ..infrastructure.tree_generator import scatter_trees
from ..infrastructure.terrain_generator import generate_river, rasterize_river_distance, River
from ..infrastructure.heightmap import generate_heightmap
from ..infrastructure.bridge_generator import generate_bridges, generate_docks
from ..infrastructure.landmark_generator import add_landmarks

@dataclass
class GenerateVillageParams:
    seed: int = 0
    width: int = 300
    height: int = 300
    tags: List[str] = None

class GenerateVillage:
    def __init__(self, params: GenerateVillageParams):
        self.params = params
        self.rng = RNG(params.seed)
        self.tags = set(params.tags or [])

    def execute(self) -> Village:
        gw = max(140, int(self.params.width))
        gh = max(140, int(self.params.height))

        density = fractal_noise(gw, gh, self.rng, octaves=4)
        heightmap = generate_heightmap(gw, gh, self.rng)

        river: River | None = None
        water_mask = np.zeros((gh, gw), dtype=bool)
        river_pts = None

        if "river" in self.tags or "coast" in self.tags:
            river = generate_river(gw, gh, self.rng)
            river_pts = river.centerline.points
            dist_to_r = rasterize_river_distance(river, gw, gh)
            water_mask = dist_to_r < river.width * 0.5

            bank = np.clip(1.0 - dist_to_r / (river.width * 1.8), 0, 1)
            density = np.maximum(density, bank * 0.9)
            density = np.where(water_mask, 0.0, density)

        # radial falloff to define village shape
        yy, xx = np.mgrid[0:gh, 0:gw]
        cx, cy = gw / 2, gh / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (min(gw, gh) / 2)
        falloff = np.clip(1.0 - r, 0, 1) ** 1.45
        density = density * falloff

        if "dense" in self.tags:
            density = np.clip(density * 1.25, 0, 1)
        if "sparse" in self.tags:
            density = density * 0.85

        major_count = 2 if "dead_end" in self.tags else 3
        if "highway" in self.tags:
            major_count = 2
        if "crossroads" in self.tags:
            major_count = 4

        minor_per_major = 7 if "dense" in self.tags else 4

        roads = generate_roads(
            density,
            self.rng,
            major_count,
            minor_per_major,
            highway=("highway" in self.tags),
        )

        density_factor = 1.35 if "dense" in self.tags else 1.0
        density_factor *= 0.8 if "sparse" in self.tags else 1.0

        houses = place_houses(roads, self.rng, density_factor=density_factor)
        houses = add_landmarks(houses, roads, self.rng, count=2 if "dense" in self.tags else 1)

        fields = generate_fields(houses, self.rng, gw, gh)

        tree_count = int(gw * gh * 0.0017)
        trees = scatter_trees(density, roads, houses, self.rng, tree_count)

        bridges = []
        docks = []
        if river_pts:
            bridges = generate_bridges(roads, river_pts, self.rng)
            docks = generate_docks(bridges, self.rng)

        v = Village(width=self.params.width, height=self.params.height, tags=list(self.tags))
        v.roads = roads
        v.houses = houses
        v.fields = fields
        v.trees = trees
        v.bridges = bridges + docks
        v.debug["grid_w"] = gw
        v.debug["grid_h"] = gh
        v.debug["heightmap_levels"] = [0.25, 0.45, 0.65, 0.8]
        v.debug["heightmap"] = heightmap  # for renderer only

        if river is not None:
            v.debug["river"] = {
                "width": river.width,
                "points": [{"x": p.x, "y": p.y} for p in river.centerline.points],
            }
        return v
