"""
Advanced Terrain Generation with Simplex Noise and Spatial Organization

This module provides a comprehensive terrain generation system that replaces
the previous noise implementation with Simplex noise for more natural terrain
patterns and integrates with Voronoi-based spatial organization.

Key Features:
- Simplex noise for improved terrain generation
- Multi-scale elevation mapping
- River generation with natural meandering
- Voronoi region integration for spatial consistency
- Distance-based terrain constraints
- Enhanced procedural detail generation

The system now uses Simplex noise which provides superior terrain characteristics
compared to traditional value noise, including reduced directional artifacts,
better clustering of features, and more natural terrain patterns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import math
from .rng import RNG
from .simplex_noise import generate_simplex_noise, SimplexNoise
from .voronoi import VoronoiRegion, VoronoiGenerator
from ..domain.geometry import Point, Polyline

@dataclass
class River:
    """
    Represents a natural river with centerline and width properties.
    
    Attributes:
        centerline: Polyline defining the river's path
        width: River width in grid units
        elevation_profile: Elevation data along the river path
        meandering_factor: Degree of natural meandering
    """
    centerline: Polyline
    width: float
    elevation_profile: Optional[np.ndarray] = None
    meandering_factor: float = 0.18

@dataclass
class TerrainConstraints:
    """
    Defines constraints and parameters for terrain generation.
    
    Attributes:
        elevation_scale: Scale factor for elevation variation
        water_level: Threshold for water areas
        river_probability: Probability of river generation
        max_elevation: Maximum elevation value
        min_elevation: Minimum elevation value
        voronoi_integration: Whether to use Voronoi-based elevation
    """
    elevation_scale: float = 1.0
    water_level: float = 0.3
    river_probability: float = 0.7
    max_elevation: float = 1.0
    min_elevation: float = 0.0
    voronoi_integration: bool = True

class AdvancedTerrainGenerator:
    """
    Advanced terrain generator using Simplex noise and Voronoi spatial organization.
    
    This system provides sophisticated terrain generation capabilities:
    - Simplex noise for natural elevation patterns
    - Multi-frequency terrain analysis
    - River and water body generation
    - Voronoi region-based elevation control
    - Distance-based terrain constraints
    - Enhanced procedural detail
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: TerrainConstraints = None
    ):
        """
        Initialize the advanced terrain generator.
        
        Args:
            width: Width of the terrain area
            height: Height of the terrain area
            constraints: Terrain generation constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or TerrainConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def generate_elevation_map(
        self, 
        rng: RNG, 
        voronoi_regions: Dict[str, VoronoiRegion] = None
    ) -> np.ndarray:
        """
        Generate a comprehensive elevation map using Simplex noise.
        
        This method creates natural-looking terrain by combining multiple
        scales of Simplex noise with regional variation control.
        
        Args:
            rng: Random number generator
            voronoi_regions: Optional Voronoi regions for spatial control
            
        Returns:
            2D array of elevation values in range [0, 1]
        """
        # Generate multi-scale Simplex noise
        elevation = self._generate_base_elevation(rng)
        
        # Apply Voronoi-based regional variation if enabled
        if voronoi_regions and self.constraints.voronoi_integration:
            elevation = self._apply_voronoi_elevation_modulation(
                elevation, voronoi_regions, rng
            )
        
        # Add procedural details and refinement
        elevation = self._add_terrain_details(elevation, rng)
        
        # Apply global constraints and normalization
        elevation = self._normalize_terrain(elevation)
        
        return elevation
    
    def _generate_base_elevation(self, rng: RNG) -> np.ndarray:
        """
        Generate base elevation using multi-scale Simplex noise.
        
        Args:
            rng: Random number generator
            
        Returns:
            Base elevation map
        """
        # Create Simplex noise generator
        simplex = SimplexNoise(seed=rng.seed)
        
        # Generate elevation with multiple scales
        elevation = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Large-scale features (continents, major landforms)
        large_scale = simplex.fractal_noise_2d(
            width=self.width,
            height=self.height,
            scale=0.3,
            octaves=5,
            persistence=0.6,
            lacunarity=2.2,
            offset_x=rng.uniform(0, 1000),
            offset_y=rng.uniform(0, 1000)
        )
        
        # Medium-scale features (hills, valleys)
        medium_scale = simplex.fractal_noise_2d(
            width=self.width,
            height=self.height,
            scale=1.0,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0,
            offset_x=rng.uniform(0, 500),
            offset_y=rng.uniform(0, 500)
        )
        
        # Small-scale features (local terrain variation)
        small_scale = simplex.fractal_noise_2d(
            width=self.width,
            height=self.height,
            scale=3.0,
            octaves=3,
            persistence=0.4,
            lacunarity=2.0,
            offset_x=rng.uniform(0, 250),
            offset_y=rng.uniform(0, 250)
        )
        
        # Combine scales with appropriate weighting
        elevation = (
            large_scale * 0.5 +      # 50% large scale
            medium_scale * 0.35 +    # 35% medium scale
            small_scale * 0.15       # 15% small scale
        )
        
        # Apply radial falloff for natural village shapes
        elevation = self._apply_radial_falloff(elevation)
        
        return elevation
    
    def _apply_voronoi_elevation_modulation(
        self,
        elevation: np.ndarray,
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> np.ndarray:
        """
        Modulate elevation based on Voronoi regions for spatial consistency.
        
        Args:
            elevation: Base elevation map
            voronoi_regions: Voronoi regions for spatial organization
            rng: Random number generator
            
        Returns:
            Elevation map with regional variation
        """
        modulated = elevation.copy()
        
        # Apply regional elevation characteristics
        for region in voronoi_regions.values():
            # Create a mask for this region
            region_mask = self._create_region_mask(region)
            
            # Determine regional characteristics
            if region.region_type == "center":
                # Central regions tend to be slightly elevated
                elevation_offset = rng.gauss(0.05, 0.02)
            elif region.region_type == "edge":
                # Edge regions tend to be lower
                elevation_offset = rng.gauss(-0.03, 0.015)
            else:
                # Intermediate regions maintain base elevation
                elevation_offset = rng.gauss(0, 0.01)
            
            # Apply the elevation modulation
            modulated[region_mask] += elevation_offset
        
        # Smooth transitions between regions
        modulated = self._smooth_regional_transitions(modulated, voronoi_regions)
        
        return modulated
    
    def _create_region_mask(self, region: VoronoiRegion) -> np.ndarray:
        """
        Create a boolean mask for a Voronoi region.
        
        Args:
            region: Voronoi region
            
        Returns:
            Boolean mask indicating pixels within the region
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Create a point-in-polygon mask for the region vertices
        for y in range(self.height):
            for x in range(self.width):
                point = Point(x + 0.5, y + 0.5)
                if self._point_in_region_polygon(point, region.vertices):
                    mask[y, x] = True
        
        return mask
    
    def _point_in_region_polygon(self, point: Point, vertices: List[Point]) -> bool:
        """
        Check if a point is inside a polygon using ray casting.
        
        Args:
            point: Point to test
            vertices: Polygon vertices
            
        Returns:
            True if point is inside the polygon
        """
        n = len(vertices)
        if n < 3:
            return False
        
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = vertices[i].x, vertices[i].y
            xj, yj = vertices[j].x, vertices[j].y
            
            if ((yi > point.y) != (yj > point.y)) and \
               (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _smooth_regional_transitions(
        self, 
        elevation: np.ndarray, 
        regions: Dict[str, VoronoiRegion]
    ) -> np.ndarray:
        """
        Smooth transitions between Voronoi regions to avoid artifacts.
        
        Args:
            elevation: Elevation map with regional variation
            regions: Voronoi regions
            
        Returns:
            Smoothed elevation map
        """
        # Apply a gentle Gaussian blur to smooth transitions
        from scipy import ndimage
        
        smoothed = ndimage.gaussian_filter(elevation, sigma=2.0)
        
        # Blend original and smoothed based on distance from region boundaries
        # This maintains sharp features while smoothing transitions
        
        return 0.7 * elevation + 0.3 * smoothed
    
    def _add_terrain_details(self, elevation: np.ndarray, rng: RNG) -> np.ndarray:
        """
        Add fine-scale terrain details for natural appearance.
        
        Args:
            elevation: Base elevation map
            rng: Random number generator
            
        Returns:
            Elevation map with added details
        """
        detailed = elevation.copy()
        
        # Add micro-scale variation for natural texture
        simplex = SimplexNoise(seed=rng.seed + 1000)
        micro_detail = simplex.fractal_noise_2d(
            width=self.width,
            height=self.height,
            scale=8.0,
            octaves=2,
            persistence=0.3,
            lacunarity=2.5,
            offset_x=rng.uniform(0, 100),
            offset_y=rng.uniform(0, 100)
        )
        
        # Blend micro detail with low weight
        detailed += micro_detail * 0.05
        
        # Add erosion-like effects for valleys
        detailed = self._apply_erosion_effects(detailed)
        
        return detailed
    
    def _apply_erosion_effects(self, elevation: np.ndarray) -> np.ndarray:
        """
        Apply erosion-like effects to create realistic valley structures.
        
        Args:
            elevation: Base elevation map
            
        Returns:
            Elevation map with erosion effects
        """
        # Identify potential valley areas (low elevation regions)
        valley_mask = elevation < 0.4
        
        if not np.any(valley_mask):
            return elevation
        
        # Apply slight smoothing to valley areas
        from scipy import ndimage
        
        # Create a copy for modification
        eroded = elevation.copy()
        
        # Apply stronger smoothing to valley areas
        smoothed_valleys = ndimage.gaussian_filter(
            elevation * valley_mask, sigma=1.5
        )
        
        # Blend based on valley presence
        eroded[valley_mask] = (
            0.3 * elevation[valley_mask] + 
            0.7 * smoothed_valleys[valley_mask]
        )
        
        return eroded
    
    def _apply_radial_falloff(self, elevation: np.ndarray) -> np.ndarray:
        """
        Apply radial falloff to create natural village shapes.
        
        Args:
            elevation: Base elevation map
            
        Returns:
            Elevation map with radial falloff applied
        """
        # Create coordinate grids
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        
        # Calculate distance from center
        cx, cy = self.width / 2, self.height / 2
        distance_from_center = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        normalized_distance = distance_from_center / (min(self.width, self.height) / 2)
        
        # Apply falloff curve
        falloff = np.clip(1.0 - normalized_distance, 0, 1) ** 1.45
        
        # Apply falloff to elevation
        elevation = elevation * falloff
        
        return elevation
    
    def _normalize_terrain(self, elevation: np.ndarray) -> np.ndarray:
        """
        Apply global terrain constraints and normalization.
        
        Args:
            elevation: Raw elevation map
            
        Returns:
            Normalized elevation map
        """
        # Apply constraints
        elevation = np.clip(
            elevation, 
            self.constraints.min_elevation, 
            self.constraints.max_elevation
        )
        
        # Normalize to [0, 1] range
        min_val = np.min(elevation)
        max_val = np.max(elevation)
        
        if max_val > min_val:
            elevation = (elevation - min_val) / (max_val - min_val)
        
        # Apply final constraints
        elevation = np.clip(elevation, 0, 1)
        
        return elevation
    
    def generate_river(
        self, 
        elevation: np.ndarray, 
        rng: RNG,
       voronoi_regions: Dict[str, VoronoiRegion] = None
    ) -> Optional[River]:
        """
        Generate a natural meandering river using elevation data.
        
        This method creates rivers that follow natural elevation gradients
        and integrate with Voronoi regions for spatial consistency.
        
        Args:
            elevation: Elevation map for river pathfinding
            rng: Random number generator
            voronoi_regions: Optional Voronoi regions for river constraints
            
        Returns:
            Generated river or None if generation fails
        """
        # Check river probability
        if rng.rand() > self.constraints.river_probability:
            return None
        
        try:
            # Generate river parameters
            river_width = rng.uniform(8.0, 14.0)
            meander_strength = self.constraints.water_level * 0.3
            
            # Generate river path using elevation gradients
            river_path = self._generate_river_path(
                elevation, voronoi_regions, rng
            )
            
            if len(river_path) < 5:  # Minimum path length
                return None
            
            # Create centerline polyline
            centerline = Polyline(river_path)
            centerline = centerline.chaikin_smooth(3)
            
            # Calculate elevation profile along river
            elevation_profile = self._calculate_river_elevation_profile(
                centerline, elevation
            )
            
            return River(
                centerline=centerline,
                width=river_width,
                elevation_profile=elevation_profile,
                meandering_factor=meander_strength
            )
            
        except Exception as e:
            # Handle generation failures gracefully
            print(f"River generation failed: {e}")
            return None
    
    def _generate_river_path(
        self,
        elevation: np.ndarray,
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> List[Point]:
        """
        Generate a natural river path following elevation gradients.
        
        Args:
            elevation: Elevation map
            voronoi_regions: Voronoi regions for path constraints
            rng: Random number generator
            
        Returns:
            List of points forming the river path
        """
        # Strategy: Start from high elevation and follow gradient downhill
        
        # Find high elevation start points
        high_elevation_threshold = np.percentile(elevation, 80)
        high_points = np.where(elevation >= high_elevation_threshold)
        
        if len(high_points[0]) == 0:
            # Fallback: start from random position
            start_x = rng.randint(10, self.width - 10)
            start_y = rng.randint(10, self.height - 10)
        else:
            # Choose random high elevation point
            idx = rng.randint(len(high_points[0]))
            start_x = high_points[1][idx]
            start_y = high_points[0][idx]
        
        # Generate river path
        path = self._trace_river_from_start(
            start_x, start_y, elevation, voronoi_regions, rng
        )
        
        return path
    
    def _trace_river_from_start(
        self,
        start_x: int,
        start_y: int,
        elevation: np.ndarray,
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> List[Point]:
        """
        Trace a river path from a starting point following elevation gradients.
        
        Args:
            start_x, start_y: Starting coordinates
            elevation: Elevation map
            voronoi_regions: Voronoi regions
            rng: Random number generator
            
        Returns:
            River path points
        """
        path = [Point(start_x + 0.5, start_y + 0.5)]
        current_x, current_y = start_x, start_y
        
        steps = 0
        max_steps = min(self.width, self.height) * 2
        
        while steps < max_steps:
            steps += 1
            
            # Get elevation and gradient at current position
            current_elevation = self._sample_elevation(current_x, current_y, elevation)
            
            # Find steepest descent direction
            gradient_direction = self._calculate_gradient_direction(
                current_x, current_y, elevation
            )
            
            # Add some randomness to river meandering
            angle = math.atan2(gradient_direction.y, gradient_direction.x)
            angle += rng.gauss(0, 0.3)  # Add meandering
            
            # Calculate next step
            step_size = rng.uniform(2.0, 4.0)
            new_x = current_x + math.cos(angle) * step_size
            new_y = current_y + math.sin(angle) * step_size
            
            # Check bounds
            if (new_x < 5 or new_y < 5 or 
                new_x >= self.width - 5 or new_y >= self.height - 5):
                break
            
            # Check if we've reached low elevation (potential river end)
            new_elevation = self._sample_elevation(int(new_x), int(new_y), elevation)
            if new_elevation < self.constraints.water_level * 0.5:
                path.append(Point(new_x + 0.5, new_y + 0.5))
                break
            
            # Update position
            current_x, current_y = new_x, new_y
            path.append(Point(current_x + 0.5, current_y + 0.5))
        
        return path
    
    def _sample_elevation(self, x: int, y: int, elevation: np.ndarray) -> float:
        """
        Sample elevation at a coordinate with bounds checking.
        
        Args:
            x, y: Coordinates to sample
            elevation: Elevation map
            
        Returns:
            Elevation value
        """
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return float(elevation[y, x])
    
    def _calculate_gradient_direction(
        self, 
        x: int, 
        y: int, 
        elevation: np.ndarray
    ) -> Point:
        """
        Calculate the gradient direction at a point.
        
        Args:
            x, y: Coordinates
            elevation: Elevation map
            
        Returns:
            Gradient direction vector
        """
        # Sample surrounding elevations
        dx = (self._sample_elevation(x + 1, y, elevation) - 
              self._sample_elevation(x - 1, y, elevation))
        dy = (self._sample_elevation(x, y + 1, elevation) - 
              self._sample_elevation(x, y - 1, elevation))
        
        # Return direction of steepest descent (negative gradient)
        length = math.hypot(dx, dy)
        if length > 0:
            return Point(-dx / length, -dy / length)
        else:
            return Point(0, -1)  # Default downward direction
    
    def _calculate_river_elevation_profile(
        self,
        centerline: Polyline,
        elevation: np.ndarray
    ) -> np.ndarray:
        """
        Calculate elevation profile along a river centerline.
        
        Args:
            centerline: River centerline
            elevation: Elevation map
            
        Returns:
            Array of elevation values along the river
        """
        profile = []
        
        for point in centerline.points:
            x, y = int(point.x), int(point.y)
            elev = self._sample_elevation(x, y, elevation)
            profile.append(elev)
        
        return np.array(profile)
    
    def rasterize_river_distance(self, river: River) -> np.ndarray:
        """
        Create a distance field from a river centerline.
        
        This method calculates the distance from each pixel to the river
        for use in terrain generation and water masking.
        
        Args:
            river: River object
            
        Returns:
            2D array of distances to river
        """
        dist = np.full((self.height, self.width), np.inf, dtype=np.float32)
        points = river.centerline.points
        
        # Process each point in the centerline
        for point in points:
            ix, iy = int(point.x), int(point.y)
            
            # Calculate local window for efficiency
            window_size = int(river.width * 3)
            x0 = max(0, ix - window_size)
            x1 = min(self.width, ix + window_size + 1)
            y0 = max(0, iy - window_size)
            y1 = min(self.height, iy + window_size + 1)
            
            # Create coordinate grids for this window
            yy, xx = np.mgrid[y0:y1, x0:x1]
            
            # Calculate distances
            dloc = np.sqrt((xx - point.x) ** 2 + (yy - point.y) ** 2)
            
            # Update distance field
            dist[y0:y1, x0:x1] = np.minimum(dist[y0:y1, x0:x1], dloc)
        
        return dist

# Legacy compatibility functions
def generate_river(width: int, height: int, rng: RNG, meander: float = 0.18) -> River:
    """
    Legacy compatibility function for river generation.
    
    This maintains the original interface while using the new terrain system.
    
    Args:
        width: River generation width
        height: River generation height
        rng: Random number generator
        meander: River meandering factor
        
    Returns:
        Generated river
    """
    generator = AdvancedTerrainGenerator(width, height)
    
    # Generate a simple elevation map for river generation
    elevation = generator._generate_base_elevation(rng)
    
    # Create river with legacy parameters
    river = generator.generate_river(elevation, rng)
    
    if river:
        river.meandering_factor = meander
    
    return river or River(Polyline([]), width=0)

def rasterize_river_distance(river: River, width: int, height: int) -> np.ndarray:
    """
    Legacy compatibility function for river distance rasterization.
    
    Args:
        river: River object
        width: Output width
        height: Output height
        
    Returns:
        Distance field to river
    """
    generator = AdvancedTerrainGenerator(width, height)
    return generator.rasterize_river_distance(river)

def generate_heightmap(width: int, height: int, rng: RNG) -> np.ndarray:
    """
    Legacy compatibility function for heightmap generation.
    
    Args:
        width: Map width
        height: Map height
        rng: Random number generator
        
    Returns:
        Generated heightmap
    """
    generator = AdvancedTerrainGenerator(width, height)
    return generator.generate_elevation_map(rng)
