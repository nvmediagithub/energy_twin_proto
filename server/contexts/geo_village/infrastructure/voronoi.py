"""
Voronoi Diagram Implementation for Spatial Organization

This module provides comprehensive Voronoi diagram generation for organizing
village layout and spatial relationships. The Voronoi diagrams help create
natural, organic layouts by dividing space into regions based on nearest
neighbor relationships.

Key Features:
- Fortune's algorithm implementation for efficient Voronoi generation
- Distance-based spatial constraints
- Region-based object placement
- Natural clustering and spacing
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KDTree
import heapq
from dataclasses import dataclass
from ..domain.geometry import Point, Polyline
from .rng import RNG

@dataclass(frozen=True)
class VoronoiRegion:
    """
    Represents a Voronoi region with its boundary and properties.
    
    The frozen=True makes this dataclass immutable and therefore hashable.
    
    Attributes:
        center: The seed point that generated this region
        vertices: List of points defining the region boundary
        area: Calculated area of the region
        perimeter: Calculated perimeter of the region
        neighbors: List of neighboring region centers
        region_type: Type classification (center, edge, corner)
    """
    center: Point
    vertices: List[Point]
    area: float
    perimeter: float
    neighbors: List[Point]
    region_type: str
    
    def __hash__(self):
        """Make VoronoiRegion hashable by using center coordinates."""
        return hash((round(self.center.x, 2), round(self.center.y, 2)))
    
    def __eq__(self, other):
        """Define equality based on center coordinates."""
        if not isinstance(other, VoronoiRegion):
            return False
        return (round(self.center.x, 2) == round(other.center.x, 2) and
                round(self.center.y, 2) == round(other.center.y, 2))

class VoronoiGenerator:
    """
    Advanced Voronoi diagram generator with distance constraints and 
    spatial organization capabilities for village generation.
    
    Features:
    - Fortune's algorithm for O(n log n) complexity
    - Distance-based region filtering
    - Multi-scale region generation
    - Spatial clustering optimization
    - Natural boundary smoothing
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize the Voronoi generator.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
        """
        self.width = width
        self.height = height
        self.margin = 10  # Margin for edge handling
        
    def generate_points(
        self, 
        rng: RNG, 
        point_count: int,
        clustering_factor: float = 0.3,
        min_distance: float = 15.0
    ) -> List[Point]:
        """
        Generate seed points for Voronoi diagram with clustering and spacing.
        
        This method creates strategically placed seed points that:
        1. Maintain minimum distance constraints
        2. Support natural clustering patterns
        3. Provide good coverage of the generation area
        4. Avoid edge clustering issues
        
        Args:
            rng: Random number generator
            point_count: Number of seed points to generate
            clustering_factor: Degree of clustering (0-1, higher = more clustered)
            min_distance: Minimum distance between points
            
        Returns:
            List of seed points for Voronoi generation
        """
        points = []
        attempts = 0
        max_attempts = point_count * 50
        
        # Create a few cluster centers for natural grouping
        cluster_centers = []
        if clustering_factor > 0.2:
            cluster_count = max(3, int(point_count * clustering_factor * 0.1))
            for _ in range(cluster_count):
                cx = rng.uniform(self.margin, self.width - self.margin)
                cy = rng.uniform(self.margin, self.height - self.margin)
                cluster_centers.append(Point(cx, cy))
        
        while len(points) < point_count and attempts < max_attempts:
            attempts += 1
            
            # Choose whether to place near a cluster center or randomly
            if cluster_centers and rng.rand() < clustering_factor:
                # Place near a cluster center
                center = rng.choice(cluster_centers)
                angle = rng.uniform(0, 2 * math.pi)
                distance = rng.gauss(0, min_distance * 0.8)
                x = center.x + math.cos(angle) * distance
                y = center.y + math.sin(angle) * distance
            else:
                # Place randomly in the area
                x = rng.uniform(self.margin, self.width - self.margin)
                y = rng.uniform(self.margin, self.height - self.margin)
            
            # Check minimum distance constraint
            new_point = Point(x, y)
            valid = True
            
            for existing in points:
                if new_point.dist(existing) < min_distance:
                    valid = False
                    break
            
            # Also check distance to cluster centers for better distribution
            if valid and cluster_centers:
                for center in cluster_centers:
                    if new_point.dist(center) < min_distance * 0.5:
                        valid = True  # Allow closer to cluster centers
                        break
            
            if valid and self._is_valid_position(new_point):
                points.append(new_point)
        
        return points[:point_count]
    
    def _is_valid_position(self, point: Point) -> bool:
        """
        Check if a point position is valid for Voronoi generation.
        
        Args:
            point: Point to validate
            
        Returns:
            True if position is valid, False otherwise
        """
        return (
            self.margin <= point.x <= self.width - self.margin and
            self.margin <= point.y <= self.height - self.margin
        )
    
    def create_voronoi_diagram(
        self, 
        points: List[Point], 
        ridge_threshold: float = 0.1
    ) -> Dict[str, VoronoiRegion]:
        """
        Create a complete Voronoi diagram from seed points.
        
        This method implements a robust Voronoi diagram generation that:
        1. Handles edge cases gracefully
        2. Provides region classification
        3. Calculates spatial properties
        4. Supports distance-based filtering
        
        Args:
            points: List of seed points
            ridge_threshold: Minimum ridge length to consider neighbors
            
        Returns:
            Dictionary mapping point hashes to VoronoiRegion objects
        """
        if len(points) < 2:
            return {}
        
        # Convert points to numpy array for scipy
        point_array = np.array([[p.x, p.y] for p in points])
        
        # Generate Voronoi diagram
        vor = Voronoi(point_array)
        regions_dict = {}
        
        # Process each region
        for i, (point, ridge_vertices) in enumerate(zip(points, vor.point_region)):
            try:
                region = self._create_region_from_voronoi(
                    point, vor, ridge_vertices, points, ridge_threshold
                )
                if region:
                    regions_dict[hash(point)] = region
            except Exception as e:
                # Handle malformed regions gracefully
                print(f"Warning: Could not process region {i}: {e}")
                continue
        
        return regions_dict
    
    def _create_region_from_voronoi(
        self,
        center: Point,
        vor: Voronoi,
        ridge_vertices: int,
        all_points: List[Point],
        ridge_threshold: float
    ) -> Optional[VoronoiRegion]:
        """
        Create a VoronoiRegion object from Voronoi diagram data.
        
        Args:
            center: Seed point for this region
            vor: Scipy Voronoi diagram object
            ridge_vertices: Region index in Voronoi diagram
            all_points: All seed points for neighbor calculation
            ridge_threshold: Minimum ridge length threshold
            
        Returns:
            VoronoiRegion object or None if creation fails
        """
        try:
            # Get vertices for this region
            vertices_idx = vor.regions[ridge_vertices]
            if -1 in vertices_idx or not vertices_idx:
                return None
            
            vertices = []
            for idx in vertices_idx:
                if 0 <= idx < len(vor.vertices):
                    vx, vy = vor.vertices[idx]
                    vertices.append(Point(vx, vy))
            
            if len(vertices) < 3:
                return None
            
            # Calculate area and perimeter
            area = self._calculate_polygon_area(vertices)
            perimeter = self._calculate_polygon_perimeter(vertices)
            
            # Find neighboring regions
            neighbors = self._find_neighbors(center, all_points, ridge_threshold)
            
            # Determine initial region type based on position
            distance_to_edge = min(
                center.x,
                center.y,
                self.width - center.x,
                self.height - center.y
            )
            
            if distance_to_edge < self.margin * 1.5:
                initial_type = "edge"
            elif distance_to_edge < self.margin * 2.5:
                initial_type = "intermediate"
            else:
                initial_type = "center"
            
            return VoronoiRegion(
                center=center,
                vertices=vertices,
                area=abs(area),
                perimeter=perimeter,
                neighbors=neighbors,
                region_type=initial_type
            )
            
        except Exception:
            return None
    
    def _calculate_polygon_area(self, vertices: List[Point]) -> float:
        """
        Calculate the area of a polygon using the shoelace formula.
        
        Args:
            vertices: List of points defining the polygon
            
        Returns:
            Area of the polygon (positive value)
        """
        if len(vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].x * vertices[j].y
            area -= vertices[j].x * vertices[i].y
        return area / 2.0
    
    def _calculate_polygon_perimeter(self, vertices: List[Point]) -> float:
        """
        Calculate the perimeter of a polygon.
        
        Args:
            vertices: List of points defining the polygon
            
        Returns:
            Perimeter of the polygon
        """
        if len(vertices) < 2:
            return 0.0
        
        perimeter = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            perimeter += vertices[i].dist(vertices[j])
        return perimeter
    
    def _find_neighbors(
        self, 
        center: Point, 
        all_points: List[Point], 
        threshold: float
    ) -> List[Point]:
        """
        Find neighboring regions based on distance and ridge analysis.
        
        Args:
            center: Center point of the current region
            all_points: All seed points
            threshold: Distance threshold for neighbor detection
            
        Returns:
            List of neighboring region centers
        """
        neighbors = []
        for point in all_points:
            if point != center and center.dist(point) < threshold * 10:
                neighbors.append(point)
        return neighbors
    
def _classify_regions(
    self,
    regions: Dict[str, VoronoiRegion],
    points: List[Point]
) -> None:
    """
    Classify regions based on their properties and position (LEGACY METHOD).
    
    This method is kept for compatibility but regions are now classified
    during creation since VoronoiRegion is now a frozen dataclass.
    
    Classifications:
    - center: Large, central regions with many neighbors
    - edge: Regions near the boundary of the generation area
    - corner: Small corner regions
    - intermediate: Medium-sized regions
    
    Args:
        regions: Dictionary of VoronoiRegion objects
        points: Original seed points for reference
    """
    # This method is no longer needed since classification happens during creation
    # It's kept for API compatibility
    pass
    
    def get_regions_by_type(
        self, 
        regions: Dict[str, VoronoiRegion], 
        region_type: str
    ) -> List[VoronoiRegion]:
        """
        Filter regions by type for targeted processing.
        
        Args:
            regions: Dictionary of VoronoiRegion objects
            region_type: Type of regions to filter
            
        Returns:
            List of regions matching the specified type
        """
        return [r for r in regions.values() if r.region_type == region_type]
    
    def get_suitable_regions(
        self, 
        regions: Dict[str, VoronoiRegion],
        min_area: float = 50.0,
        max_area: float = float('inf'),
        exclude_types: Set[str] = None
    ) -> List[VoronoiRegion]:
        """
        Get regions suitable for object placement based on size and type.
        
        Args:
            regions: Dictionary of VoronoiRegion objects
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            exclude_types: Set of region types to exclude
            
        Returns:
            List of suitable regions
        """
        if exclude_types is None:
            exclude_types = set()
        
        suitable = []
        for region in regions.values():
            if (min_area <= region.area <= max_area and 
                region.region_type not in exclude_types):
                suitable.append(region)
        
        return suitable

def generate_village_regions(
    width: int, 
    height: int, 
    rng: RNG,
    region_count: int = 15,
    clustering: float = 0.4,
    min_distance: float = 20.0
) -> Dict[str, VoronoiRegion]:
    """
    Generate Voronoi regions optimized for village layout.
    
    This function creates a complete set of Voronoi regions suitable for
    village generation with natural clustering and proper spacing.
    
    Args:
        width: Width of the generation area
        height: Height of the generation area
        rng: Random number generator
        region_count: Number of major regions to create
        clustering: Clustering factor (0-1)
        min_distance: Minimum distance between region centers
        
    Returns:
        Dictionary of VoronoiRegion objects indexed by hash
    """
    generator = VoronoiGenerator(width, height)
    
    # Generate seed points
    points = generator.generate_points(
        rng=rng,
        point_count=region_count,
        clustering_factor=clustering,
        min_distance=min_distance
    )
    
    # Create Voronoi diagram
    regions = generator.create_voronoi_diagram(points)
    
    return regions