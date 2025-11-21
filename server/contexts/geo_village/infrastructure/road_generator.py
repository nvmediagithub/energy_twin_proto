"""
Advanced Road Generation System with Voronoi Spatial Organization

This module provides a completely redesigned road generation system that leverages
Voronoi diagrams for natural, spatially-organized road networks. The new system
replaces the previous MST-based approach with Voronoi-region-guided road placement
for more realistic village layouts.

Key Features:
- Voronoi-based road placement for spatial organization
- Distance-constrained road spacing
- Natural road hierarchies (major, minor, local)
- Road connectivity optimization
- Regional road density control
- Integration with house and landmark placement

The system now uses Voronoi regions as the primary organizing principle for road
networks, ensuring roads follow natural spatial boundaries and maintain proper
distances between all infrastructure elements.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from .rng import RNG
from .voronoi import VoronoiRegion, VoronoiGenerator
from ..domain.geometry import Point, Polyline
from ..domain.models import Road

@dataclass
class RoadNetworkConstraints:
    """
    Defines constraints and parameters for road network generation.
    
    Attributes:
        min_road_spacing: Minimum distance between parallel roads
        max_road_length: Maximum length for individual road segments
        connectivity_threshold: Minimum connectivity between regions
        road_density_factor: Overall road density multiplier
        voronoi_integration: Whether to use Voronoi-based placement
        hierarchical_roads: Whether to maintain road hierarchy
    """
    min_road_spacing: float = 12.0
    max_road_length: float = 80.0
    connectivity_threshold: float = 0.6
    road_density_factor: float = 1.0
    voronoi_integration: bool = True
    hierarchical_roads: bool = True

@dataclass
class RoadSegment:
    """
    Represents a road segment before final processing.
    
    Attributes:
        start_point: Starting point of the segment
        end_point: Ending point of the segment
        road_type: Type of road (major, minor, local)
        priority: Priority for inclusion in final network
        region_id: ID of the region this segment belongs to
    """
    start_point: Point
    end_point: Point
    road_type: str
    priority: float
    region_id: str

class VoronoiRoadGenerator:
    """
    Advanced road generator using Voronoi spatial organization.
    
    This system creates realistic road networks by leveraging Voronoi diagrams
    for spatial organization and natural road placement patterns.
    
    Features:
    - Voronoi-region-based road placement
    - Hierarchical road network generation
    - Distance-constrained spacing
    - Connectivity optimization
    - Natural road clustering
    - Regional road density control
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: RoadNetworkConstraints = None
    ):
        """
        Initialize the Voronoi road generator.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
            constraints: Road network generation constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or RoadNetworkConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def generate_road_network(
        self,
        density: np.ndarray,
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG,
        major_count: int,
        minor_per_major: int,
        highway: bool = False
    ) -> List[Road]:
        """
        Generate a complete road network using Voronoi spatial organization.
        
        This method creates a hierarchical road network that:
        1. Connects Voronoi regions naturally
        2. Maintains proper spacing between roads
        3. Follows natural spatial boundaries
        4. Provides optimal connectivity
        
        Args:
            density: Density map for road placement
            voronoi_regions: Voronoi regions for spatial organization
            rng: Random number generator
            major_count: Number of major roads to generate
            minor_per_major: Minor roads per major road
            highway: Whether to generate highway-style roads
            
        Returns:
            List of generated roads
        """
        # Filter and rank regions for road placement
        suitable_regions_with_priority = self._filter_regions_for_roads(voronoi_regions, density)
        
        # Extract just the regions (without priority) for processing
        suitable_regions = [region for region, priority in suitable_regions_with_priority]
        
        # Generate hierarchical road segments
        major_roads = self._generate_major_roads(suitable_regions, major_count, rng)
        minor_roads = self._generate_minor_roads(
            suitable_regions, major_roads, minor_per_major, rng
        )
        local_roads = self._generate_local_roads(suitable_regions, density, rng)
        
        # Combine and optimize road network
        all_roads = major_roads + minor_roads + local_roads
        
        # Apply spacing constraints
        optimized_roads = self._apply_spacing_constraints(all_roads)
        
        # Convert segments to final road objects
        final_roads = self._create_road_objects(optimized_roads, density, rng)
        
        # Apply final optimization and smoothing
        final_roads = self._finalize_road_network(final_roads)
        
        return final_roads
    
    def _filter_regions_for_roads(
        self,
        regions: Dict[str, VoronoiRegion],
        density: np.ndarray
    ) -> List[Tuple[VoronoiRegion, float]]:
        """
        Filter and rank Voronoi regions suitable for road placement.
        
        Args:
            regions: All available Voronoi regions
            density: Density map for placement guidance
            
        Returns:
            List of (region, priority) tuples ranked by road placement priority
        """
        suitable = []
        
        for region in regions.values():
            # Skip very small regions
            if region.area < 50:
                continue
            
            # Skip corner regions for main roads
            if region.region_type == "corner":
                continue
            
            # Calculate road placement priority
            priority = self._calculate_region_road_priority(region, density)
            
            if priority > 0.3:  # Minimum threshold
                suitable.append((region, priority))
        
        # Sort by priority
        suitable.sort(key=lambda x: x[1], reverse=True)
        
        return suitable
    
    def _calculate_region_road_priority(
        self,
        region: VoronoiRegion,
        density: np.ndarray
    ) -> float:
        """
        Calculate priority score for road placement in a region.
        
        Args:
            region: Voronoi region
            density: Density map
            
        Returns:
            Priority score (0-1)
        """
        # Base priority from region type
        type_priority = {
            "center": 1.0,
            "intermediate": 0.8,
            "edge": 0.6,
            "corner": 0.3
        }.get(region.region_type, 0.5)
        
        # Calculate average density in region
        region_mask = self._create_region_mask(region)
        if np.any(region_mask):
            avg_density = np.mean(density[region_mask])
        else:
            avg_density = 0.5
        
        # Prefer regions with good density
        density_score = avg_density
        
        # Consider region size (medium-sized regions are preferred)
        optimal_area = self.width * self.height * 0.01
        size_score = 1.0 - abs(region.area - optimal_area) / optimal_area
        size_score = max(0, size_score)
        
        # Combine scores
        total_priority = (
            type_priority * 0.4 +
            density_score * 0.4 +
            size_score * 0.2
        )
        
        return min(1.0, total_priority)
    
    def _create_region_mask(self, region: VoronoiRegion) -> np.ndarray:
        """
        Create a boolean mask for a Voronoi region.
        
        Args:
            region: Voronoi region
            
        Returns:
            Boolean mask indicating pixels within the region
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Simple approximation using region center and bounding box
        center_x, center_y = int(region.center.x), int(region.center.y)
        radius = int(math.sqrt(region.area) * 0.5)
        
        x0 = max(0, center_x - radius)
        x1 = min(self.width, center_x + radius)
        y0 = max(0, center_y - radius)
        y1 = min(self.height, center_y + radius)
        
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        mask[y0:y1, x0:x1] = distances <= radius
        
        return mask
    
    def _generate_major_roads(
        self,
        regions: List[VoronoiRegion],
        count: int,
        rng: RNG
    ) -> List[RoadSegment]:
        """
        Generate major roads connecting important regions.
        
        Args:
            regions: Suitable regions for road placement
            count: Number of major roads to generate
            rng: Random number generator
            
        Returns:
            List of major road segments
        """
        major_roads = []
        
        # Select high-priority regions for major road endpoints
        selected_regions = regions[:max(count * 2, len(regions) // 2)]
        
        # Generate major roads as connections between region centers
        for i in range(min(count, len(selected_regions) // 2)):
            region1 = selected_regions[i]
            region2 = selected_regions[(i + 1) % len(selected_regions)]
            
            # Create road segment between regions
            road = self._create_region_connection_road(
                region1, region2, "major", rng
            )
            if road:
                major_roads.append(road)
        
        # Add highway-style roads if requested
        if count > len(major_roads):
            highway_roads = self._generate_highway_roads(selected_regions, rng)
            major_roads.extend(highway_roads)
        
        return major_roads
    
    def _create_region_connection_road(
        self,
        region1: VoronoiRegion,
        region2: VoronoiRegion,
        road_type: str,
        rng: RNG
    ) -> Optional[RoadSegment]:
        """
        Create a road segment connecting two Voronoi regions.
        
        Args:
            region1: First region
            region2: Second region
            road_type: Type of road to create
            rng: Random number generator
            
        Returns:
            Road segment connecting the regions
        """
        # Calculate connection points (prefer region boundaries)
        start_point = self._find_region_boundary_point(region1, region2)
        end_point = self._find_region_boundary_point(region2, region1)
        
        if not start_point or not end_point:
            # Fallback to region centers
            start_point = region1.center
            end_point = region2.center
        
        # Calculate priority based on region types and distances
        priority = self._calculate_connection_priority(region1, region2)
        
        return RoadSegment(
            start_point=start_point,
            end_point=end_point,
            road_type=road_type,
            priority=priority,
            region_id=f"{hash(region1)}_{hash(region2)}"
        )
    
    def _find_region_boundary_point(
        self,
        source_region: VoronoiRegion,
        target_region: VoronoiRegion
    ) -> Optional[Point]:
        """
        Find an optimal boundary point for connecting to another region.
        
        Args:
            source_region: Region to find boundary point for
            target_region: Target region for connection
            
        Returns:
            Boundary point for connection
        """
        # Find point on source region boundary closest to target
        direction = Point(
            target_region.center.x - source_region.center.x,
            target_region.center.y - source_region.center.y
        )
        
        length = math.hypot(direction.x, direction.y)
        if length == 0:
            return source_region.center
        
        direction = Point(direction.x / length, direction.y / length)
        
        # Find intersection with region boundary
        for vertex in source_region.vertices:
            # Check if this vertex is in the direction of the target
            vertex_direction = Point(
                vertex.x - source_region.center.x,
                vertex.y - source_region.center.y
            )
            vertex_length = math.hypot(vertex_direction.x, vertex_direction.y)
            
            if vertex_length > 0:
                vertex_direction = Point(
                    vertex_direction.x / vertex_length,
                    vertex_direction.y / vertex_length
                )
                
                # Check if vertex is roughly in the direction of target
                dot_product = (
                    direction.x * vertex_direction.x + 
                    direction.y * vertex_direction.y
                )
                
                if dot_product > 0.5:  # Within 60 degrees
                    return vertex
        
        # Fallback to center if no suitable boundary found
        return source_region.center
    
    def _calculate_connection_priority(
        self,
        region1: VoronoiRegion,
        region2: VoronoiRegion
    ) -> float:
        """
        Calculate priority for connecting two regions.
        
        Args:
            region1: First region
            region2: Second region
            
        Returns:
            Connection priority score
        """
        # Base priority from region types
        type_scores = {
            "center": 1.0,
            "intermediate": 0.8,
            "edge": 0.6,
            "corner": 0.4
        }
        
        score1 = type_scores.get(region1.region_type, 0.5)
        score2 = type_scores.get(region2.region_type, 0.5)
        
        # Distance factor (prefer medium distances)
        distance = region1.center.dist(region2.center)
        optimal_distance = min(self.width, self.height) * 0.3
        distance_factor = 1.0 - abs(distance - optimal_distance) / optimal_distance
        distance_factor = max(0.1, distance_factor)
        
        # Neighbor factor (prefer regions that are already neighbors)
        neighbor_factor = 1.0 if region2.center in region1.neighbors else 0.7
        
        # Combine factors
        total_priority = (score1 + score2) * 0.4 + distance_factor * 0.4 + neighbor_factor * 0.2
        
        return min(1.0, total_priority)
    
    def _generate_highway_roads(
        self,
        regions: List[VoronoiRegion],
        rng: RNG
    ) -> List[RoadSegment]:
        """
        Generate highway-style roads across the map.
        
        Args:
            regions: Available regions
            rng: Random number generator
            
        Returns:
            List of highway road segments
        """
        highway_roads = []
        
        # Create horizontal and vertical highways
        if len(regions) >= 4:
            # Horizontal highway
            top_region = regions[0]
            bottom_region = regions[-1]
            highway_h = self._create_region_connection_road(
                top_region, bottom_region, "major", rng
            )
            if highway_h:
                highway_roads.append(highway_h)
            
            # Vertical highway
            left_region = regions[len(regions) // 2]
            right_region = regions[(len(regions) // 2 + 1) % len(regions)]
            highway_v = self._create_region_connection_road(
                left_region, right_region, "major", rng
            )
            if highway_v:
                highway_roads.append(highway_v)
        
        return highway_roads
    
    def _generate_minor_roads(
        self,
        regions: List[VoronoiRegion],
        major_roads: List[RoadSegment],
        count_per_major: int,
        rng: RNG
    ) -> List[RoadSegment]:
        """
        Generate minor roads connecting to major roads.
        
        Args:
            regions: Available regions
            major_roads: Major road network
            count_per_major: Minor roads per major road
            rng: Random number generator
            
        Returns:
            List of minor road segments
        """
        minor_roads = []
        
        for major_road in major_roads:
            # Generate minor roads branching from this major road
            branches = self._generate_road_branches(
                major_road, regions, count_per_major, rng
            )
            minor_roads.extend(branches)
        
        return minor_roads
    
    def _generate_road_branches(
        self,
        major_road: RoadSegment,
        regions: List[VoronoiRegion],
        count: int,
        rng: RNG
    ) -> List[RoadSegment]:
        """
        Generate road branches from a major road.
        
        Args:
            major_road: Major road to branch from
            regions: Available regions
            count: Number of branches to generate
            rng: Random number generator
            
        Returns:
            List of branch road segments
        """
        branches = []
        
        # Find suitable regions for branching
        start_point = major_road.start_point
        available_regions = [r for r in regions if r.center.dist(start_point) > 20]
        
        for _ in range(min(count, len(available_regions))):
            if not available_regions:
                break
            
            # Select random region for branching
            region = rng.choice(available_regions)
            available_regions.remove(region)
            
            # Create branch road
            branch = RoadSegment(
                start_point=start_point,
                end_point=region.center,
                road_type="minor",
                priority=rng.uniform(0.3, 0.7),
                region_id=f"branch_{hash(region)}"
            )
            
            branches.append(branch)
        
        return branches
    
    def _generate_local_roads(
        self,
        regions: List[VoronoiRegion],
        density: np.ndarray,
        rng: RNG
    ) -> List[RoadSegment]:
        """
        Generate local roads within regions.
        
        Args:
            regions: Available regions
            density: Density map for placement guidance
            rng: Random number generator
            
        Returns:
            List of local road segments
        """
        local_roads = []
        
        for region in regions:
            # Generate local roads based on region density and size
            if region.area > 200 and region.region_type in ["center", "intermediate"]:
                region_local_roads = self._generate_intra_region_roads(
                    region, density, rng
                )
                local_roads.extend(region_local_roads)
        
        return local_roads
    
    def _generate_intra_region_roads(
        self,
        region: VoronoiRegion,
        density: np.ndarray,
        rng: RNG
    ) -> List[RoadSegment]:
        """
        Generate roads within a single region.
        
        Args:
            region: Voronoi region
            density: Density map
            rng: Random number generator
            
        Returns:
            List of intra-region road segments
        """
        local_roads = []
        
        # Generate 1-3 local roads per suitable region
        num_local = rng.randint(1, 4)
        
        for _ in range(num_local):
            # Create random start and end points within region
            start_point = self._random_point_in_region(region, rng)
            end_point = self._random_point_in_region(region, rng)
            
            if start_point and end_point and start_point.dist(end_point) > 10:
                road = RoadSegment(
                    start_point=start_point,
                    end_point=end_point,
                    road_type="minor",
                    priority=rng.uniform(0.2, 0.5),
                    region_id=f"local_{hash(region)}"
                )
                local_roads.append(road)
        
        return local_roads
    
    def _random_point_in_region(
        self,
        region: VoronoiRegion,
        rng: RNG
    ) -> Optional[Point]:
        """
        Generate a random point within a Voronoi region.
        
        Args:
            region: Voronoi region
            rng: Random number generator
            
        Returns:
            Random point within the region
        """
        # Simple approach: random point within bounding box
        # This is an approximation - for more accuracy, use point-in-polygon
        
        min_x = min(v.x for v in region.vertices)
        max_x = max(v.x for v in region.vertices)
        min_y = min(v.y for v in region.vertices)
        max_y = max(v.y for v in region.vertices)
        
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        
        return Point(x, y)
    
    def _apply_spacing_constraints(
        self,
        roads: List[RoadSegment]
    ) -> List[RoadSegment]:
        """
        Apply spacing constraints to road segments.
        
        Args:
            roads: Road segments to process
            
        Returns:
            Filtered road segments meeting spacing constraints
        """
        if len(roads) <= 1:
            return roads
        
        # Sort by priority
        roads.sort(key=lambda r: r.priority, reverse=True)
        
        filtered_roads = []
        
        for road in roads:
            # Check distance from existing roads
            valid = True
            
            for existing_road in filtered_roads:
                min_distance = self._calculate_road_distance(road, existing_road)
                
                if min_distance < self.constraints.min_road_spacing:
                    valid = False
                    break
            
            if valid:
                filtered_roads.append(road)
        
        return filtered_roads
    
    def _calculate_road_distance(
        self,
        road1: RoadSegment,
        road2: RoadSegment
    ) -> float:
        """
        Calculate minimum distance between two road segments.
        
        Args:
            road1: First road segment
            road2: Second road segment
            
        Returns:
            Minimum distance between the segments
        """
        # Calculate distances between all endpoints
        distances = [
            road1.start_point.dist(road2.start_point),
            road1.start_point.dist(road2.end_point),
            road1.end_point.dist(road2.start_point),
            road1.end_point.dist(road2.end_point)
        ]
        
        return min(distances)
    
    def _create_road_objects(
        self,
        road_segments: List[RoadSegment],
        density: np.ndarray,
        rng: RNG
    ) -> List[Road]:
        """
        Convert road segments to final Road objects.
        
        Args:
            road_segments: Road segments to convert
            density: Density map for road parameters
            rng: Random number generator
            
        Returns:
            List of Road objects
        """
        roads = []
        
        for segment in road_segments:
            # Create polyline with some noise for natural appearance
            polyline = self._create_noisy_polyline(segment, rng)
            
            # Create road object
            road = Road(polyline=polyline, kind=segment.road_type)
            roads.append(road)
        
        return roads
    
    def _create_noisy_polyline(
        self,
        segment: RoadSegment,
        rng: RNG
    ) -> Polyline:
        """
        Create a polyline with natural noise for a road segment.
        
        Args:
            segment: Road segment
            rng: Random number generator
            
        Returns:
            Noisy polyline
        """
        points = [segment.start_point]
        
        # Add intermediate points with noise
        steps = max(3, int(segment.start_point.dist(segment.end_point) / 15))
        
        for i in range(1, steps):
            t = i / steps
            
            # Linear interpolation with noise
            x = (segment.start_point.x * (1 - t) + 
                 segment.end_point.x * t + 
                 rng.gauss(0, 2.0))
            y = (segment.start_point.y * (1 - t) + 
                 segment.end_point.y * t + 
                 rng.gauss(0, 2.0))
            
            points.append(Point(x, y))
        
        points.append(segment.end_point)
        
        # Create and smooth polyline
        polyline = Polyline(points)
        return polyline.chaikin_smooth(2)
    
    def _finalize_road_network(self, roads: List[Road]) -> List[Road]:
        """
        Apply final optimization and smoothing to the road network.
        
        Args:
            roads: Initial road network
            
        Returns:
            Optimized and smoothed road network
        """
        # Remove duplicate or very short roads
        filtered_roads = self._remove_redundant_roads(roads)
        
        # Apply final smoothing
        for road in filtered_roads:
            road.polyline = road.polyline.chaikin_smooth(1)
        
        return filtered_roads
    
    def _remove_redundant_roads(self, roads: List[Road]) -> List[Road]:
        """
        Remove redundant or overly short roads.
        
        Args:
            roads: Roads to filter
            
        Returns:
            Filtered list of roads
        """
        filtered = []
        
        for road in roads:
            # Check if road is too short
            total_length = 0
            points = road.polyline.points
            
            for i in range(len(points) - 1):
                total_length += points[i].dist(points[i + 1])
            
            if total_length < 8:  # Minimum road length
                continue
            
            # Check for significant overlap with existing roads
            redundant = False
            for existing_road in filtered:
                if self._roads_significantly_overlap(road, existing_road):
                    redundant = True
                    break
            
            if not redundant:
                filtered.append(road)
        
        return filtered
    
    def _roads_significantly_overlap(
        self,
        road1: Road,
        road2: Road
    ) -> bool:
        """
        Check if two roads significantly overlap.
        
        Args:
            road1: First road
            road2: Second road
            
        Returns:
            True if roads significantly overlap
        """
        # Simple overlap check based on endpoint proximity
        overlap_threshold = 10.0
        
        for p1 in road1.polyline.points:
            for p2 in road2.polyline.points:
                if p1.dist(p2) < overlap_threshold:
                    return True
        
        return False

# Legacy compatibility functions
def generate_roads(
    density: np.ndarray,
    rng: RNG,
    major_count: int,
    minor_per_major: int,
    highway: bool = False
) -> List[Road]:
    """
    Legacy compatibility function for road generation.
    
    This maintains the original interface while using the new Voronoi-based system.
    
    Args:
        density: Density map for road placement
        rng: Random number generator
        major_count: Number of major roads
        minor_per_major: Minor roads per major road
        highway: Whether to generate highway-style roads
        
    Returns:
        Generated road network
    """
    # Estimate map dimensions from density
    height, width = density.shape
    
    # Generate Voronoi regions for spatial organization
    voronoi_regions = generate_village_regions(
        width=width,
        height=height,
        rng=rng,
        region_count=max(10, major_count * 3),
        clustering=0.4,
        min_distance=25.0
    )
    
    # Create generator and generate roads
    generator = VoronoiRoadGenerator(width, height)
    return generator.generate_road_network(
        density=density,
        voronoi_regions=voronoi_regions,
        rng=rng,
        major_count=major_count,
        minor_per_major=minor_per_major,
        highway=highway
    )

def _pick_density_peaks(density: np.ndarray, rng: RNG, k: int) -> List[Point]:
    """
    Legacy compatibility function for density peak picking.
    
    Args:
        density: Density map
        rng: Random number generator
        k: Number of peaks to pick
        
    Returns:
        List of peak points
    """
    from scipy import ndimage
    
    # Find local maxima
    local_maxima = ndimage.maximum_filter(density, size=5) == density
    peaks = np.where(local_maxima)
    
    if len(peaks[0]) == 0:
        # Fallback to random points
        return [Point(rng.uniform(0, density.shape[1]), rng.uniform(0, density.shape[0])) 
                for _ in range(k)]
    
    # Sort by density value and select top k
    peak_indices = list(zip(peaks[1], peaks[0]))  # (x, y)
    peak_values = [density[y, x] for x, y in peak_indices]
    
    sorted_indices = np.argsort(peak_values)[-k:]
    selected_peaks = [peak_indices[i] for i in sorted_indices]
    
    return [Point(x + 0.5, y + 0.5) for x, y in selected_peaks]

def _mst(points: List[Point]) -> List[Tuple[int, int]]:
    """
    Legacy compatibility function for minimum spanning tree.
    
    Args:
        points: Points for MST calculation
        
    Returns:
        List of edge tuples
    """
    from scipy.spatial.distance import pdist, squareform
    
    if len(points) < 2:
        return []
    
    # Calculate distance matrix
    point_array = np.array([[p.x, p.y] for p in points])
    distances = squareform(pdist(point_array))
    
    # Simple Prim's algorithm
    n = len(points)
    in_tree = [False] * n
    in_tree[0] = True
    edges = []
    
    for _ in range(n - 1):
        min_dist = float('inf')
        u, v = -1, -1
        
        for i in range(n):
            if in_tree[i]:
                for j in range(n):
                    if not in_tree[j] and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        u, v = i, j
        
        if u != -1 and v != -1:
            in_tree[v] = True
            edges.append((u, v))
    
    return edges

def _noisy_path(a: Point, b: Point, rng: RNG, steps: int = 20, jitter: float = 4.0) -> Polyline:
    """
    Legacy compatibility function for noisy path generation.
    
    Args:
        a: Start point
        b: End point
        rng: Random number generator
        steps: Number of steps
        jitter: Jitter amount
        
    Returns:
        Noisy polyline
    """
    points = [a]
    
    for s in range(1, steps):
        t = s / steps
        x = a.x * (1 - t) + b.x * t
        y = a.y * (1 - t) + b.y * t
        
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy) + 1e-9
        nx, ny = -dy / length, dx / length
        
        amplitude = math.sin(t * math.pi)
        x += nx * rng.gauss(0, jitter) * amplitude
        y += ny * rng.gauss(0, jitter) * amplitude
        
        points.append(Point(x, y))
    
    points.append(b)
    return Polyline(points).chaikin_smooth(2)
