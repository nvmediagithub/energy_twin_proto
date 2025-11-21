"""
Advanced House Placement System with Square Buildings and Distance Constraints

This module provides a completely redesigned house placement system that replaces
the previous rectangular design with square buildings and implements sophisticated
distance-based spacing algorithms.

Key Features:
- Square building design with uniform dimensions
- Voronoi-based spatial organization
- Distance-based collision avoidance
- Natural clustering patterns
- Improved architectural variety
- Regional placement optimization

The system uses Voronoi diagrams to organize space and ensure proper spacing
between all buildings while maintaining natural village patterns.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from dataclasses import dataclass
from .rng import RNG
from .voronoi import VoronoiRegion, VoronoiGenerator
from ..domain.geometry import Point, Polyline
from ..domain.models import Road, House

@dataclass
class HousePlacementConstraints:
    """
    Defines spatial constraints for house placement.
    
    Attributes:
        min_distance: Minimum distance between any two houses
        road_offset: Minimum distance from roads
        voronoi_constraint: Whether to use Voronoi-based placement
        max_houses_per_region: Maximum houses per Voronoi region
        clustering_factor: Degree of natural clustering (0-1)
    """
    min_distance: float = 8.0
    road_offset: float = 3.0
    voronoi_constraint: bool = True
    max_houses_per_region: int = 3
    clustering_factor: float = 0.3

class SquareHousePlacer:
    """
    Advanced house placement system for square buildings with distance constraints.
    
    This system replaces the previous rectangular house design with square buildings
    and implements sophisticated spatial constraints using Voronoi diagrams for
    natural, organic village layouts.
    
    Features:
    - Square building design for architectural consistency
    - Voronoi-based spatial organization
    - Multi-level distance constraints
    - Natural clustering with controlled density
    - Road proximity optimization
    - Regional capacity management
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: HousePlacementConstraints = None
    ):
        """
        Initialize the square house placement system.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
            constraints: Spatial placement constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or HousePlacementConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def place_houses_in_regions(
        self,
        regions: Dict[str, VoronoiRegion],
        roads: List[Road],
        rng: RNG,
        target_density: float = 0.8
    ) -> List[House]:
        """
        Place houses using Voronoi regions for optimal spatial organization.
        
        This method places houses strategically within Voronoi regions to create
        natural village layouts with proper spacing and clustering.
        
        Args:
            regions: Voronoi regions for spatial organization
            roads: Road network for placement reference
            rng: Random number generator
            target_density: Target density factor (0-1)
            
        Returns:
            List of placed square houses
        """
        houses = []
        
        # Filter suitable regions for house placement
        suitable_regions = self._filter_suitable_regions(
            regions, target_density
        )
        
        # Sort regions by suitability (prefer central, medium-sized regions)
        suitable_regions.sort(key=lambda x: self._calculate_region_suitability(x[0]), reverse=True)
        
        # Place houses in each suitable region
        for region, capacity in suitable_regions:
            region_houses = self._place_houses_in_region(
                region, roads, rng, capacity
            )
            houses.extend(region_houses)
        
        # Apply global spacing constraints
        houses = self._apply_global_constraints(houses)
        
        return houses
    
    def _filter_suitable_regions(
        self,
        regions: Dict[str, VoronoiRegion],
        target_density: float
    ) -> List[Tuple[VoronoiRegion, int]]:
        """
        Filter and rank Voronoi regions suitable for house placement.
        
        Args:
            regions: All available Voronoi regions
            target_density: Target density factor
            
        Returns:
            List of tuples (region, capacity) sorted by suitability
        """
        suitable = []
        
        for region in regions.values():
            # Skip unsuitable region types
            if region.region_type in ["corner", "edge"]:
                continue
            
            # Check area constraints
            min_area = self.width * self.height * 0.002  # Minimum viable area
            max_area = self.width * self.height * 0.05   # Maximum area to avoid oversized regions
            
            if region.area < min_area or region.area > max_area:
                continue
            
            # Consider density factor
            density_factor = min(1.0, target_density / 0.8)
            
            # Calculate capacity based on region size
            region_capacity = max(1, int(region.area / (min_area * 2)))
            if region_capacity > self.constraints.max_houses_per_region:
                region_capacity = self.constraints.max_houses_per_region
            
            # Apply density factor
            actual_capacity = max(1, int(region_capacity * density_factor))
            
            # Store as tuple instead of modifying frozen dataclass
            suitable.append((region, actual_capacity))
        
        return suitable
    
    def _calculate_region_suitability(self, region: VoronoiRegion) -> float:
        """
        Calculate suitability score for a region for house placement.
        
        Higher scores indicate better regions for house placement.
        
        Args:
            region: Voronoi region to evaluate
            
        Returns:
            Suitability score (0-1)
        """
        # Prefer interior regions over edge regions
        edge_penalty = 0.0
        if region.region_type == "edge":
            edge_penalty = 0.3
        elif region.region_type == "corner":
            edge_penalty = 0.6
        
        # Prefer medium-sized regions
        optimal_area = self.width * self.height * 0.01
        size_score = 1.0 - abs(region.area - optimal_area) / optimal_area
        
        # Prefer regions with more neighbors (better connectivity)
        neighbor_score = min(1.0, len(region.neighbors) / 6.0)
        
        # Combine scores with weights
        total_score = (size_score * 0.4 + neighbor_score * 0.3 + 
                      (1.0 - edge_penalty) * 0.3)
        
        return max(0.0, total_score)
    
    def _place_houses_in_region(
        self,
        region: VoronoiRegion,
        roads: List[Road],
        rng: RNG,
        capacity: int = 1
    ) -> List[House]:
        """
        Place houses within a single Voronoi region.
        
        Args:
            region: Voronoi region for house placement
            roads: Road network
            rng: Random number generator
            capacity: Number of houses to place in this region
            
        Returns:
            List of houses placed in this region
        """
        houses = []
        
        # Get road proximity points in this region
        road_proximity_points = self._get_road_proximity_points(region, roads)
        
        # Place houses based on region characteristics
        if capacity == 1:
            # Single house placement
            house = self._place_single_house(region, road_proximity_points, rng)
            if house:
                houses.append(house)
        else:
            # Multiple houses with clustering
            houses = self._place_multiple_houses(
                region, road_proximity_points, capacity, rng
            )
        
        return houses
    
    def _get_road_proximity_points(
        self,
        region: VoronoiRegion,
        roads: List[Road]
    ) -> List[Point]:
        """
        Find points in the region near roads for strategic house placement.
        
        Args:
            region: Voronoi region to analyze
            roads: Road network
            
        Returns:
            List of road-proximate points in the region
        """
        proximity_points = []
        
        for road in roads:
            for point in road.polyline.points:
                if self._point_in_region(point, region):
                    # Check if point is within reasonable distance from region center
                    center_dist = point.dist(region.center)
                    if center_dist < region.perimeter * 0.3:
                        proximity_points.append(point)
        
        return proximity_points
    
    def _point_in_region(self, point: Point, region: VoronoiRegion) -> bool:
        """
        Check if a point is inside a Voronoi region using ray casting.
        
        Args:
            point: Point to test
            region: Voronoi region
            
        Returns:
            True if point is inside the region
        """
        vertices = region.vertices
        n = len(vertices)
        
        # Ray casting algorithm
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
    
    def _place_single_house(
        self,
        region: VoronoiRegion,
        road_points: List[Point],
        rng: RNG
    ) -> Optional[House]:
        """
        Place a single house in a region with optimal positioning.
        
        Args:
            region: Voronoi region for placement
            road_points: Road-proximate points
            rng: Random number generator
            
        Returns:
            Placed house or None if placement fails
        """
        # Prefer placement near roads
        if road_points and rng.rand() < 0.7:
            # Place near a road point with some offset
            road_point = rng.choice(road_points)
            direction = self._calculate_optimal_direction(road_point, region)
            
            # Calculate position with offset from road
            offset_distance = rng.uniform(3.0, 8.0)
            angle = math.atan2(direction.y, direction.x) + rng.gauss(0, 0.3)
            
            x = road_point.x + math.cos(angle) * offset_distance
            y = road_point.y + math.sin(angle) * offset_distance
        else:
            # Place randomly within region
            x = rng.uniform(
                region.center.x - region.area * 0.01,
                region.center.x + region.area * 0.01
            )
            y = rng.uniform(
                region.center.y - region.area * 0.01,
                region.center.y + region.area * 0.01
            )
        
        # Create square house
        center = Point(x, y)
        
        # Ensure the position is valid
        if not self._is_valid_house_position(center, region, []):
            return None
        
        # Generate square house parameters
        house = self._create_square_house(center, region, rng)
        return house
    
    def _place_multiple_houses(
        self,
        region: VoronoiRegion,
        road_points: List[Point],
        count: int,
        rng: RNG
    ) -> List[House]:
        """
        Place multiple houses in a region with clustering.
        
        Args:
            region: Voronoi region for placement
            road_points: Road-proximate points
            count: Number of houses to place
            rng: Random number generator
            
        Returns:
            List of placed houses
        """
        houses = []
        
        # Determine placement strategy based on region size and count
        if len(road_points) > 0 and count <= 2:
            # Linear placement near roads
            houses = self._place_linear_cluster(
                region, road_points, count, rng
            )
        else:
            # Radial or grid placement within region
            houses = self._place_radial_cluster(
                region, count, rng
            )
        
        return houses
    
    def _place_linear_cluster(
        self,
        region: VoronoiRegion,
        road_points: List[Point],
        count: int,
        rng: RNG
    ) -> List[House]:
        """
        Place houses in a linear cluster near roads.
        
        Args:
            region: Voronoi region
            road_points: Road reference points
            count: Number of houses
            rng: Random number generator
            
        Returns:
            List of placed houses
        """
        houses = []
        base_point = rng.choice(road_points)
        
        for i in range(count):
            # Calculate position along a line with spacing
            direction = self._calculate_optimal_direction(base_point, region)
            angle = math.atan2(direction.y, direction.x)
            
            # Add perpendicular offset for variety
            perp_angle = angle + math.pi/2 if i % 2 == 0 else angle - math.pi/2
            
            # Calculate house position
            distance = self.constraints.min_distance * (i + 1) * 0.8
            x = base_point.x + math.cos(perp_angle) * distance + rng.gauss(0, 1.0)
            y = base_point.y + math.sin(perp_angle) * distance + rng.gauss(0, 1.0)
            
            center = Point(x, y)
            
            # Validate position
            if self._is_valid_house_position(center, region, houses):
                house = self._create_square_house(center, region, rng)
                if house:
                    houses.append(house)
        
        return houses
    
    def _place_radial_cluster(
        self,
        region: VoronoiRegion,
        count: int,
        rng: RNG
    ) -> List[House]:
        """
        Place houses in a radial cluster around region center.
        
        Args:
            region: Voronoi region
            count: Number of houses
            rng: Random number generator
            
        Returns:
            List of placed houses
        """
        houses = []
        
        # Calculate cluster center (prefer region center but adjust for roads)
        cluster_center = region.center
        
        # Place houses in a spiral or circular pattern
        for i in range(count):
            # Calculate position using golden angle for natural spacing
            golden_angle = math.pi * (3 - math.sqrt(5))  # Golden angle in radians
            angle = i * golden_angle
            
            # Calculate distance from center with some randomness
            base_distance = self.constraints.min_distance * math.sqrt(i + 1)
            distance = base_distance + rng.gauss(0, base_distance * 0.2)
            
            # Add position with natural variation
            x = cluster_center.x + math.cos(angle) * distance + rng.gauss(0, 1.5)
            y = cluster_center.y + math.sin(angle) * distance + rng.gauss(0, 1.5)
            
            center = Point(x, y)
            
            # Validate position
            if self._is_valid_house_position(center, region, houses):
                house = self._create_square_house(center, region, rng)
                if house:
                    houses.append(house)
        
        return houses
    
    def _calculate_optimal_direction(self, point: Point, region: VoronoiRegion) -> Point:
        """
        Calculate optimal direction from a point toward region center.
        
        Args:
            point: Starting point
            region: Target region
            
        Returns:
            Direction vector toward region center
        """
        dx = region.center.x - point.x
        dy = region.center.y - point.y
        length = math.hypot(dx, dy)
        
        if length > 0:
            return Point(dx / length, dy / length)
        else:
            return Point(0, 0)
    
    def _is_valid_house_position(
        self,
        center: Point,
        region: VoronoiRegion,
        existing_houses: List[House]
    ) -> bool:
        """
        Validate if a house position meets all constraints.
        
        Args:
            center: Proposed house center
            region: Target Voronoi region
            existing_houses: Already placed houses
            
        Returns:
            True if position is valid
        """
        # Check if position is within bounds
        if (center.x < 2 or center.y < 2 or 
            center.x > self.width - 2 or center.y > self.height - 2):
            return False
        
        # Check if position is within the target region
        if not self._point_in_region(center, region):
            return False
        
        # Check distance from existing houses
        for house in existing_houses:
            if center.dist(house.center) < self.constraints.min_distance:
                return False
        
        return True
    
    def _create_square_house(
        self,
        center: Point,
        region: VoronoiRegion,
        rng: RNG
    ) -> House:
        """
        Create a square house with consistent dimensions and orientation.
        
        Args:
            center: House center position
            region: Containing Voronoi region
            rng: Random number generator
            
        Returns:
            Configured square house
        """
        # Generate consistent square dimensions
        base_size = rng.uniform(3.0, 5.5)
        
        # Adjust size based on region characteristics
        if region.region_type == "center":
            size_multiplier = rng.uniform(1.2, 1.6)  # Larger houses in center
        elif region.region_type == "edge":
            size_multiplier = rng.uniform(0.8, 1.1)  # Smaller houses at edges
        else:
            size_multiplier = rng.uniform(0.9, 1.3)  # Medium houses elsewhere
        
        size = base_size * size_multiplier
        
        # Calculate orientation based on roads and region layout
        orientation = self._calculate_house_orientation(center, region, rng)
        
        # Determine side based on orientation and random factor
        side = "left" if rng.rand() < 0.5 else "right"
        
        return House(
            center=center,
            width=size,
            height=size,  # Square building: width == height
            rotation=orientation,
            side=side
        )
    
    def _calculate_house_orientation(
        self,
        center: Point,
        region: VoronoiRegion,
        rng: RNG
    ) -> float:
        """
        Calculate optimal house orientation based on spatial context.
        
        Args:
            center: House center position
            region: Containing region
            rng: Random number generator
            
        Returns:
            House rotation angle in radians
        """
        # Default random orientation with slight bias
        base_orientation = rng.uniform(0, 2 * math.pi)
        
        # Add orientation bias toward region center for visual harmony
        to_center = math.atan2(
            region.center.y - center.y,
            region.center.x - center.x
        )
        
        # Blend random orientation with center direction
        blend_factor = rng.uniform(0.2, 0.4)  # 20-40% toward center
        orientation = base_orientation * (1 - blend_factor) + to_center * blend_factor
        
        return orientation
    
    def _apply_global_constraints(self, houses: List[House]) -> List[House]:
        """
        Apply global spacing constraints and remove conflicting houses.
        
        Args:
            houses: Initially placed houses
            
        Returns:
            Filtered list of houses meeting all constraints
        """
        if len(houses) <= 1:
            return houses
        
        # Sort houses by suitability (larger houses first)
        houses.sort(key=lambda h: h.width, reverse=True)
        
        filtered_houses = []
        for house in houses:
            # Check distance from all previously accepted houses
            valid = True
            for accepted in filtered_houses:
                if house.center.dist(accepted.center) < self.constraints.min_distance:
                    valid = False
                    break
            
            if valid:
                filtered_houses.append(house)
        
        return filtered_houses
    
    def place_houses_legacy(
        self,
        roads: List[Road],
        rng: RNG,
        density_factor: float = 1.0
    ) -> List[House]:
        """
        Legacy compatibility method for square houses without Voronoi.
        
        This method provides backward compatibility while using square buildings.
        
        Args:
            roads: Road network
            rng: Random number generator
            density_factor: Density multiplier
            
        Returns:
            List of placed square houses
        """
        houses = []
        constraints = HousePlacementConstraints(
            voronoi_constraint=False,
            min_distance=8.0 / density_factor
        )
        
        # Generate simple Voronoi regions for basic spatial organization
        simple_regions = self._generate_simple_regions(rng, 8)
        regions = {}
        
        for i, region_center in enumerate(simple_regions):
            region = VoronoiRegion(
                center=region_center,
                vertices=self._generate_simple_vertices(region_center),
                area=100.0,
                perimeter=40.0,
                neighbors=[],
                region_type="simple"
            )
            regions[hash(region_center)] = region
        
        # Use the new placement method
        return self.place_houses_in_regions(
            regions=regions,
            roads=roads,
            rng=rng,
            target_density=density_factor
        )
    
    def _generate_simple_regions(self, rng: RNG, count: int) -> List[Point]:
        """
        Generate simple regions for legacy compatibility.
        
        Args:
            rng: Random number generator
            count: Number of regions to generate
            
        Returns:
            List of region centers
        """
        regions = []
        margin = 20
        
        for _ in range(count):
            x = rng.uniform(margin, self.width - margin)
            y = rng.uniform(margin, self.height - margin)
            regions.append(Point(x, y))
        
        return regions
    
    def _generate_simple_vertices(self, center: Point) -> List[Point]:
        """
        Generate simple rectangular vertices around a center point.
        
        Args:
            center: Center point for the region
            
        Returns:
            List of vertices forming a rectangle
        """
        size = 15.0
        return [
            Point(center.x - size, center.y - size),
            Point(center.x + size, center.y - size),
            Point(center.x + size, center.y + size),
            Point(center.x - size, center.y + size)
        ]

# Legacy compatibility function
def place_houses(roads: List[Road], rng: RNG, density_factor: float = 1.0) -> List[House]:
    """
    Legacy compatibility function for square house placement.
    
    This function maintains the same interface as the original but now places
    square buildings with improved spacing algorithms.
    
    Args:
        roads: Road network for placement reference
        rng: Random number generator
        density_factor: Density multiplier
        
    Returns:
        List of placed square houses with proper spacing
    """
    # Estimate map dimensions based on roads
    if roads:
        # Calculate approximate bounds from roads
        all_points = [p for road in roads for p in road.polyline.points]
        if all_points:
            xs = [p.x for p in all_points]
            ys = [p.y for p in all_points]
            width = int(max(xs) - min(xs) + 40)
            height = int(max(ys) - min(ys) + 40)
        else:
            width, height = 300, 300
    else:
        width, height = 300, 300
    
    # Create placer and place houses
    placer = SquareHousePlacer(width, height)
    return placer.place_houses_legacy(roads, rng, density_factor)
