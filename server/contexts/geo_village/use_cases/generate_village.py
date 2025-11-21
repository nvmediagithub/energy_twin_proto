"""
Advanced Village Generation Use Case with Complete System Integration

This module provides a completely redesigned village generation use case that integrates
all the new advanced systems for comprehensive village creation.

INTEGRATED SYSTEMS:
- Voronoi diagram generation for spatial organization
- Simplex noise for improved terrain generation
- Square house placement with distance constraints
- Advanced road generation using Voronoi structure
- Distance-based tree generation with spatial awareness
- Voronoi-based field generation for agricultural areas
- Enhanced landmark generation with spatial constraints
- Spatial bridge generation with terrain awareness

The system now provides a unified interface for generating villages with:
- Natural spatial organization through Voronoi diagrams
- Improved procedural generation through Simplex noise
- Proper spacing and distance constraints throughout
- Regional variation and spatial coherence
- Enhanced detail and realism in all components
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from ..domain.models import Village
from ..infrastructure.rng import RNG
from ..infrastructure.voronoi import generate_village_regions, VoronoiRegion
from ..infrastructure.simplex_noise import generate_simplex_noise
from ..infrastructure.terrain_generator import AdvancedTerrainGenerator, TerrainConstraints
from ..infrastructure.road_generator import VoronoiRoadGenerator, RoadNetworkConstraints
from ..infrastructure.house_placer import SquareHousePlacer, HousePlacementConstraints
from ..infrastructure.tree_generator import AdvancedTreeGenerator, TreePlacementConstraints
from ..infrastructure.field_generator import AdvancedFieldGenerator, FieldPlacementConstraints
from ..infrastructure.landmark_generator import AdvancedLandmarkGenerator, LandmarkPlacementConstraints
from ..infrastructure.bridge_generator import AdvancedBridgeGenerator, BridgePlacementConstraints
from ..infrastructure.noise import fractal_noise  # Keep for legacy compatibility

@dataclass
class GenerateVillageParams:
    """
    Parameters for advanced village generation with comprehensive options.
    
    This enhanced parameter set allows for detailed control over all aspects
    of the village generation process, including spatial organization,
    terrain characteristics, infrastructure placement, and aesthetic options.
    
    Attributes:
        seed: Random seed for deterministic generation
        width: Village width in grid units
        height: Village height in grid units
        tags: Generation tags for specialized village types
        voronoi_regions: Number of Voronoi regions for spatial organization
        noise_scale: Scale factor for terrain noise generation
        complexity: Overall complexity factor (0.5-2.0)
        spatial_coherence: Spatial coherence factor (0-1)
    """
    seed: int = 0
    width: int = 300
    height: int = 300
    tags: List[str] = None
    
    # Advanced parameters for new systems
    voronoi_regions: int = 15
    noise_scale: float = 1.0
    complexity: float = 1.0
    spatial_coherence: float = 0.8

class AdvancedVillageGenerator:
    """
    Advanced village generator integrating all new systems.
    
    This generator provides a comprehensive village generation system that:
    1. Uses Voronoi diagrams for spatial organization
    2. Employs Simplex noise for natural terrain generation
    3. Places square buildings with proper distance constraints
    4. Generates roads using Voronoi-based spatial patterns
    5. Creates trees with distance-based spacing
    6. Forms agricultural areas using Voronoi regions
    7. Places landmarks with spatial awareness
    8. Constructs bridges with terrain and spatial integration
    
    The system ensures all components work together harmoniously while
    maintaining proper spatial relationships and natural village patterns.
    """
    
    def __init__(self, params: GenerateVillageParams):
        """
        Initialize the advanced village generator.
        
        Args:
            params: Village generation parameters
        """
        self.params = params
        self.rng = RNG(params.seed)
        self.tags = set(params.tags or [])
        
        # Initialize all advanced generators
        self._setup_generators()
        
    def _setup_generators(self) -> None:
        """Initialize all advanced generators with appropriate constraints."""
        
        # Terrain generator with Simplex noise
        terrain_constraints = TerrainConstraints(
            elevation_scale=self.params.noise_scale,
            voronoi_integration=True,
            river_probability=0.7 if "river" in self.tags or "coast" in self.tags else 0.3
        )
        self.terrain_generator = AdvancedTerrainGenerator(
            self.params.width, 
            self.params.height, 
            terrain_constraints
        )
        
        # Road generator with Voronoi integration
        road_constraints = RoadNetworkConstraints(
            voronoi_integration=True,
            hierarchical_roads=True,
            road_density_factor=self.params.complexity
        )
        self.road_generator = VoronoiRoadGenerator(
            self.params.width, 
            self.params.height, 
            road_constraints
        )
        
        # House placer with square buildings and distance constraints
        house_constraints = HousePlacementConstraints(
            min_distance=8.0 * self.params.complexity,
            voronoi_constraint=True,
            clustering_factor=0.3 * self.params.spatial_coherence
        )
        self.house_placer = SquareHousePlacer(
            self.params.width, 
            self.params.height, 
            house_constraints
        )
        
        # Tree generator with distance-based spacing
        tree_constraints = TreePlacementConstraints(
            min_tree_spacing=3.0 * self.params.complexity,
            voronoi_integration=True,
            house_buffer=6.0,
            road_buffer=3.0
        )
        self.tree_generator = AdvancedTreeGenerator(
            self.params.width, 
            self.params.height, 
            tree_constraints
        )
        
        # Field generator with Voronoi integration
        field_constraints = FieldPlacementConstraints(
            voronoi_integration=True,
            field_size_factor=self.params.complexity
        )
        self.field_generator = AdvancedFieldGenerator(
            self.params.width, 
            self.params.height, 
            field_constraints
        )
        
        # Landmark generator with spatial constraints
        landmark_constraints = LandmarkPlacementConstraints(
            voronoi_integration=True,
            landmark_hierarchy=True,
            min_landmark_spacing=25.0 * self.params.complexity
        )
        self.landmark_generator = AdvancedLandmarkGenerator(
            self.params.width, 
            self.params.height, 
            landmark_constraints
        )
        
        # Bridge generator with spatial awareness
        bridge_constraints = BridgePlacementConstraints(
            voronoi_integration=True,
            spatial_clustering=True,
            terrain_awareness=True
        )
        self.bridge_generator = AdvancedBridgeGenerator(
            self.params.width, 
            self.params.height, 
            bridge_constraints
        )
    
    def execute(self) -> Village:
        """
        Generate a complete village using all advanced systems.
        
        This method orchestrates the complete village generation process:
        1. Generate Voronoi regions for spatial organization
        2. Create terrain using Simplex noise
        3. Generate roads using Voronoi-based patterns
        4. Place houses with square design and distance constraints
        5. Generate fields using Voronoi agricultural regions
        6. Place trees with distance-based spacing
        7. Add landmarks with spatial awareness
        8. Create bridges with terrain integration
        9. Combine all elements into final village
        
        Returns:
            Complete village with all infrastructure and elements
        """
        # Step 1: Generate Voronoi regions for spatial organization
        voronoi_regions = self._generate_voronoi_regions()
        
        # Step 2: Generate terrain using Simplex noise
        terrain_data = self._generate_terrain(voronoi_regions)
        
        # Step 3: Generate density map for placement
        density = self._generate_density_map(voronoi_regions, terrain_data)
        
        # Step 4: Generate road network using Voronoi structure
        roads = self._generate_road_network(density, voronoi_regions)
        
        # Step 5: Place houses with square design and constraints
        houses = self._place_houses(roads, voronoi_regions)
        
        # Step 6: Generate fields using Voronoi agricultural regions
        fields = self._generate_fields(houses, roads, voronoi_regions)
        
        # Step 7: Place trees with distance-based spacing
        trees = self._place_trees(density, roads, houses, voronoi_regions)
        
        # Step 8: Add landmarks with spatial constraints
        houses = self._add_landmarks(houses, roads, voronoi_regions)
        
        # Step 9: Generate bridges if rivers exist
        bridges = self._generate_bridges(roads, terrain_data)
        
        # Step 10: Create final village
        village = self._create_village(
            roads=roads,
            houses=houses,
            fields=fields,
            trees=trees,
            bridges=bridges,
            terrain_data=terrain_data
        )
        
        return village
    
    def _generate_voronoi_regions(self) -> Dict[str, VoronoiRegion]:
        """
        Generate Voronoi regions for spatial organization.
        
        Returns:
            Dictionary of Voronoi regions indexed by hash
        """
        # Adjust region count based on village size and complexity
        adjusted_region_count = max(
            8, 
            int(self.params.voronoi_regions * self.params.complexity)
        )
        
        regions = generate_village_regions(
            width=self.params.width,
            height=self.params.height,
            rng=self.rng,
            region_count=adjusted_region_count,
            clustering=self.params.spatial_coherence,
            min_distance=20.0 * self.params.complexity
        )
        
        return regions
    
    def _generate_terrain(self, voronoi_regions: Dict[str, VoronoiRegion]) -> Dict:
        """
        Generate terrain using Simplex noise and spatial organization.
        
        Args:
            voronoi_regions: Voronoi regions for spatial control
            
        Returns:
            Dictionary containing elevation map and river data
        """
        # Generate elevation map using Simplex noise
        elevation = self.terrain_generator.generate_elevation_map(
            self.rng, 
            voronoi_regions
        )
        
        # Generate river if applicable
        river = None
        river_pts = None
        
        if "river" in self.tags or "coast" in self.tags:
            river = self.terrain_generator.generate_river(
                elevation, 
                self.rng, 
                voronoi_regions
            )
            if river:
                river_pts = [p for p in river.centerline.points]
        
        return {
            "elevation": elevation,
            "river": river,
            "river_points": river_pts,
            "distance_map": self.terrain_generator.rasterize_river_distance(river) if river else None
        }
    
    def _generate_density_map(
        self, 
        voronoi_regions: Dict[str, VoronoiRegion], 
        terrain_data: Dict
    ) -> np.ndarray:
        """
        Generate density map for infrastructure placement.
        
        Args:
            voronoi_regions: Voronoi regions
            terrain_data: Terrain generation data
            
        Returns:
            Density map for placement guidance
        """
        # Start with Simplex noise for natural density patterns
        density = generate_simplex_noise(
            width=self.params.width,
            height=self.params.height,
            rng=self.rng,
            scale=1.0 * self.params.noise_scale,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0
        )
        
        # Apply radial falloff for village shape
        density = self._apply_village_falloff(density)
        
        # Apply density modifications based on tags
        density = self._apply_density_modifications(density, terrain_data)
        
        return density
    
    def _apply_village_falloff(self, density: np.ndarray) -> np.ndarray:
        """
        Apply radial falloff to define village shape.
        
        Args:
            density: Base density map
            
        Returns:
            Density map with village-shaped falloff
        """
        yy, xx = np.mgrid[0:self.params.height, 0:self.params.width]
        cx, cy = self.params.width / 2, self.params.height / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (min(self.params.width, self.params.height) / 2)
        falloff = np.clip(1.0 - r, 0, 1) ** 1.45
        
        return density * falloff
    
    def _apply_density_modifications(
        self, 
        density: np.ndarray, 
        terrain_data: Dict
    ) -> np.ndarray:
        """
        Apply density modifications based on tags and terrain.
        
        Args:
            density: Base density map
            terrain_data: Terrain generation data
            
        Returns:
            Modified density map
        """
        # Apply tag-based modifications
        if "dense" in self.tags:
            density = np.clip(density * 1.25 * self.params.complexity, 0, 1)
        if "sparse" in self.tags:
            density = density * 0.85 / self.params.complexity
        
        # Apply river effects if present
        if terrain_data.get("river") and terrain_data.get("distance_map"):
            distance_map = terrain_data["distance_map"]
            river = terrain_data["river"]
            
            # Create river banks with higher density
            bank = np.clip(1.0 - distance_map / (river.width * 1.8), 0, 1)
            density = np.maximum(density, bank * 0.9)
            
            # Avoid water areas
            water_mask = distance_map < river.width * 0.5
            density = np.where(water_mask, 0.0, density)
        
        return density
    
    def _generate_road_network(
        self, 
        density: np.ndarray, 
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> List[Road]:
        """
        Generate road network using Voronoi-based spatial organization.
        
        Args:
            density: Density map for road placement
            voronoi_regions: Voronoi regions for spatial organization
            
        Returns:
            List of generated roads
        """
        # Determine road network parameters based on village characteristics
        major_count = self._calculate_major_road_count()
        minor_per_major = self._calculate_minor_roads_per_major()
        highway = "highway" in self.tags
        
        roads = self.road_generator.generate_road_network(
            density=density,
            voronoi_regions=voronoi_regions,
            rng=self.rng,
            major_count=major_count,
            minor_per_major=minor_per_major,
            highway=highway
        )
        
        return roads
    
    def _calculate_major_road_count(self) -> int:
        """
        Calculate number of major roads based on village characteristics.
        
        Returns:
            Number of major roads
        """
        base_count = 3 if "dead_end" not in self.tags else 2
        
        if "crossroads" in self.tags:
            base_count = 4
        elif "highway" in self.tags:
            base_count = 2
        
        # Adjust based on complexity
        return max(2, int(base_count * self.params.complexity))
    
    def _calculate_minor_roads_per_major(self) -> int:
        """
        Calculate minor roads per major road based on density.
        
        Returns:
            Number of minor roads per major road
        """
        base_count = 7 if "dense" in self.tags else 4
        
        # Adjust based on complexity and spatial coherence
        adjusted_count = int(base_count * self.params.complexity * self.params.spatial_coherence)
        
        return max(2, adjusted_count)
    
    def _place_houses(
        self, 
        roads: List[Road], 
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> List[House]:
        """
        Place houses with square design and distance constraints.
        
        Args:
            roads: Road network for placement reference
            voronoi_regions: Voronoi regions for spatial organization
            
        Returns:
            List of placed square houses
        """
        # Calculate target density based on village characteristics
        target_density = 1.0 if "dense" not in self.tags else 1.35
        target_density *= 0.8 if "sparse" in self.tags else 1.0
        
        houses = self.house_placer.place_houses_in_regions(
            regions=voronoi_regions,
            roads=roads,
            rng=self.rng,
            target_density=target_density
        )
        
        return houses
    
    def _generate_fields(
        self, 
        houses: List[House], 
        roads: List[Road], 
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> List[FieldPlot]:
        """
        Generate fields using Voronoi agricultural regions.
        
        Args:
            houses: House positions
            roads: Road network
            voronoi_regions: Voronoi regions for spatial organization
            
        Returns:
            List of generated field plots
        """
        fields = self.field_generator.generate_agricultural_regions(
            houses=houses,
            voronoi_regions=voronoi_regions,
            roads=roads,
            rng=self.rng,
            map_w=self.params.width,
            map_h=self.params.height
        )
        
        return fields
    
    def _place_trees(
        self, 
        density: np.ndarray, 
        roads: List[Road], 
        houses: List[House], 
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> List[Tree]:
        """
        Place trees with distance-based spacing.
        
        Args:
            density: Density map for tree placement
            roads: Road network
            houses: House positions
            voronoi_regions: Voronoi regions for spatial organization
            
        Returns:
            List of placed trees
        """
        # Calculate target tree count based on village size
        target_count = int(self.params.width * self.params.height * 0.0017 * self.params.complexity)
        
        trees = self.tree_generator.generate_tree_distribution(
            density=density,
            roads=roads,
            houses=houses,
            voronoi_regions=voronoi_regions,
            rng=self.rng,
            target_count=target_count
        )
        
        return trees
    
    def _add_landmarks(
        self, 
        houses: List[House], 
        roads: List[Road], 
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> List[House]:
        """
        Add landmarks with spatial constraints.
        
        Args:
            houses: House positions
            roads: Road network
            voronoi_regions: Voronoi regions for spatial organization
            
        Returns:
            List of houses with landmarks added
        """
        # Calculate landmark count based on village characteristics
        landmark_count = 2 if "dense" in self.tags else 1
        landmark_count = max(1, int(landmark_count * self.params.complexity))
        
        landmarks = self.landmark_generator.generate_landmarks(
            houses=houses,
            roads=roads,
            voronoi_regions=voronoi_regions,
            rng=self.rng,
            target_count=landmark_count
        )
        
        # Apply landmarks to houses (replace some houses with landmarks)
        result_houses = []
        landmark_index = 0
        
        for house in houses:
            if (landmark_index < len(landmarks) and 
                landmark_index < landmark_count and
                hasattr(house, 'meta') and house.meta.get('landmark')):
                # This house is already a landmark
                result_houses.append(landmarks[landmark_index])
                landmark_index += 1
            else:
                # Regular house
                result_houses.append(house)
        
        return result_houses
    
    def _generate_bridges(
        self, 
        roads: List[Road], 
        terrain_data: Dict
    ) -> List[Bridge]:
        """
        Generate bridges with spatial awareness and terrain integration.
        
        Args:
            roads: Road network
            terrain_data: Terrain generation data
            
        Returns:
            List of generated bridges
        """
        if not terrain_data.get("river_points"):
            return []
        
        # Generate Voronoi regions for bridge spatial organization
        bridge_regions = generate_village_regions(
            width=self.params.width,
            height=self.params.height,
            rng=self.rng,
            region_count=8,
            clustering=0.3,
            min_distance=25.0
        )
        
        bridges = self.bridge_generator.generate_bridge_network(
            roads=roads,
            river_points=terrain_data["river_points"],
            voronoi_regions=bridge_regions,
            rng=self.rng
        )
        
        # Generate docks for some bridges
        docks = self.bridge_generator._generate_additional_bridge_types(
            bridges, 
            terrain_data["river_points"], 
            roads, 
            self.rng
        )
        
        return bridges + docks
    
    def _create_village(
        self,
        roads: List[Road],
        houses: List[House],
        fields: List[FieldPlot],
        trees: List[Tree],
        bridges: List[Bridge],
        terrain_data: Dict
    ) -> Village:
        """
        Create the final village object with all components.
        
        Args:
            roads: Road network
            houses: House positions (including landmarks)
            fields: Agricultural fields
            trees: Tree positions
            bridges: Bridge positions
            terrain_data: Terrain generation data
            
        Returns:
            Complete village object
        """
        # Create village with enhanced debug information
        village = Village(
            width=self.params.width,
            height=self.params.height,
            tags=list(self.tags)
        )
        
        # Assign all infrastructure
        village.roads = roads
        village.houses = houses
        village.fields = fields
        village.trees = trees
        village.bridges = bridges
        
        # Enhanced debug information
        village.debug = {
            "grid_w": self.params.width,
            "grid_h": self.params.height,
            "heightmap_levels": [0.25, 0.45, 0.65, 0.8],
            "heightmap": terrain_data["elevation"],
            "voronoi_regions": len([r for r in [] if hasattr(r, 'region_type')]),  # Placeholder
            "generation_algorithm": "advanced_voronoi_simplex",
            "complexity_factor": self.params.complexity,
            "spatial_coherence": self.params.spatial_coherence,
            "noise_scale": self.params.noise_scale,
            "village_regions": self.params.voronoi_regions
        }
        
        # Add river information if present
        if terrain_data.get("river"):
            village.debug["river"] = {
                "width": terrain_data["river"].width,
                "points": [{"x": p.x, "y": p.y} for p in terrain_data["river"].centerline.points],
                "meandering_factor": terrain_data["river"].meandering_factor
            }
        
        return village

# Legacy compatibility function
def generate_village(params: GenerateVillageParams) -> Village:
    """
    Legacy compatibility function for village generation.
    
    This maintains the original interface while using the new advanced system.
    
    Args:
        params: Village generation parameters
        
    Returns:
        Generated village using all new systems
    """
    generator = AdvancedVillageGenerator(params)
    return generator.execute()

# Import required modules for legacy compatibility
from ..domain.models import Road, House, FieldPlot, Tree, Bridge
from ..infrastructure.rng import RNG
