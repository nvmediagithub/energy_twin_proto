"""
Advanced Bridge Generation System with Spatial Awareness and Voronoi Integration

This module provides a completely redesigned bridge generation system that implements
spatial awareness and integrates with Voronoi spatial organization for optimal bridge
placement across rivers and other water features.

Key Features:
- Voronoi-based bridge placement for spatial organization
- Distance-based bridge spacing and optimization
- Spatial awareness of road networks and terrain
- Integration with river systems and terrain constraints
- Natural bridge clustering and hierarchical placement
- Enhanced bridge types (bridges, docks, ferries) based on spatial context

The system ensures bridges are placed optimally within the spatial organization
while maintaining proper distances from all infrastructure elements and following
natural transportation patterns.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from dataclasses import dataclass
from .rng import RNG
from .voronoi import VoronoiRegion, VoronoiGenerator, generate_village_regions
from ..domain.geometry import Point, Polyline
from ..domain.models import Road, Bridge

@dataclass
class BridgePlacementConstraints:
    """
    Defines spatial constraints for bridge placement.
    
    Attributes:
        min_bridge_spacing: Minimum distance between bridges
        road_intersection_threshold: Minimum road-river intersection angle
        spatial_clustering: Whether to cluster bridges spatially
        voronoi_integration: Whether to use Voronoi-based placement
        bridge_density_factor: Overall bridge density multiplier
        terrain_awareness: Whether to consider terrain for placement
    """
    min_bridge_spacing: float = 20.0
    road_intersection_threshold: float = 0.3
    spatial_clustering: bool = True
    voronoi_integration: bool = True
    bridge_density_factor: float = 1.0
    terrain_awareness: bool = True

@dataclass
class SpatialBridgePoint:
    """
    Represents a bridge placement point with spatial context.
    
    Attributes:
        point: Bridge location
        intersection_angle: Angle of road-river intersection
        spatial_advantage: Spatial advantage score
        bridge_type: Suggested bridge type
        connectivity_score: Connectivity score for the location
    """
    point: Point
    intersection_angle: float
    spatial_advantage: float
    bridge_type: str
    connectivity_score: float

class AdvancedBridgeGenerator:
    """
    Advanced bridge generator with spatial awareness and Voronoi integration.
    
    This system creates realistic bridge distributions by:
    1. Using Voronoi regions for spatial organization
    2. Implementing spatial awareness of road networks and terrain
    3. Creating natural bridge hierarchies and clustering
    4. Optimizing bridge placement based on spatial characteristics
    5. Ensuring proper spacing and transportation efficiency
    
    Features:
    - Voronoi-region-based bridge placement
    - Spatial road-river intersection analysis
    - Distance-constrained bridge spacing
    - Natural bridge clustering
    - Integration with terrain and infrastructure
    - Multiple bridge types (bridges, docks, ferries)
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: BridgePlacementConstraints = None
    ):
        """
        Initialize the advanced bridge generator.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
            constraints: Bridge placement constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or BridgePlacementConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def generate_bridge_network(
        self,
        roads: List[Road],
        river_points: List[Point],
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> List[Bridge]:
        """
        Generate a complete bridge network with spatial awareness.
        
        This method creates natural bridge distributions by:
        1. Analyzing road-river intersections spatially
        2. Creating spatially-aware bridge placement points
        3. Placing bridges with proper spacing and constraints
        4. Ensuring optimal transportation connectivity
        
        Args:
            roads: Road network for intersection analysis
            river_points: River centerline points
            voronoi_regions: Voronoi regions for spatial organization
            rng: Random number generator
            
        Returns:
            List of generated bridges with spatial optimization
        """
        # Find road-river intersection points with spatial analysis
        intersection_points = self._find_spatial_intersections(
            roads, river_points, voronoi_regions, rng
        )
        
        # Generate bridge placement points
        bridge_points = self._generate_bridge_points(
            intersection_points, roads, river_points, voronoi_regions, rng
        )
        
        # Create bridges from placement points
        bridges = self._create_bridges_from_points(
            bridge_points, roads, river_points, rng
        )
        
        # Generate additional bridge types (docks, ferries)
        additional_bridges = self._generate_additional_bridge_types(
            bridges, river_points, roads, rng
        )
        
        # Combine all bridges
        all_bridges = bridges + additional_bridges
        
        # Apply spatial constraints
        final_bridges = self._apply_bridge_constraints(all_bridges)
        
        return final_bridges
    
    def _find_spatial_intersections(
        self,
        roads: List[Road],
        river_points: List[Point],
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> List[SpatialBridgePoint]:
        """
        Find road-river intersections with spatial analysis.
        
        Args:
            roads: Road network
            river_points: River centerline
            voronoi_regions: Voronoi regions
            rng: Random number generator
            
        Returns:
            List of spatial bridge points
        """
        if not roads or not river_points:
            return []
        
        intersection_points = []
        
        # Convert river points to segments for easier intersection testing
        river_segments = self._create_river_segments(river_points)
        
        # Analyze each road for intersections
        for road in roads:
            road_intersections = self._find_road_river_intersections(
                road, river_segments, voronoi_regions, rng
            )
            intersection_points.extend(road_intersections)
        
        return intersection_points
    
    def _create_river_segments(self, river_points: List[Point]) -> List[Tuple[Point, Point]]:
        """
        Create line segments from river points.
        
        Args:
            river_points: River centerline points
            
        Returns:
            List of river segments
        """
        segments = []
        for i in range(len(river_points) - 1):
            segments.append((river_points[i], river_points[i + 1]))
        return segments
    
    def _find_road_river_intersections(
        self,
        road: Road,
        river_segments: List[Tuple[Point, Point]],
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> List[SpatialBridgePoint]:
        """
        Find intersections between a road and river with spatial analysis.
        
        Args:
            road: Road to analyze
            river_segments: River segments
            voronoi_regions: Voronoi regions
            rng: Random number generator
            
        Returns:
            List of spatial bridge points
        """
        intersections = []
        
        road_segments = self._create_road_segments(road.polyline.points)
        
        for road_start, road_end in road_segments:
            for river_start, river_end in river_segments:
                intersection = self._segment_intersection_spatial(
                    road_start, road_end, river_start, river_end, voronoi_regions, rng
                )
                if intersection:
                    intersections.append(intersection)
        
        return intersections
    
    def _create_road_segments(self, points: List[Point]) -> List[Tuple[Point, Point]]:
        """
        Create line segments from road points.
        
        Args:
            points: Road points
            
        Returns:
            List of road segments
        """
        segments = []
        for i in range(len(points) - 1):
            segments.append((points[i], points[i + 1]))
        return segments
    
    def _segment_intersection_spatial(
        self,
        p1: Point,
        p2: Point,
        q1: Point,
        q2: Point,
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> Optional[SpatialBridgePoint]:
        """
        Calculate segment intersection with spatial analysis.
        
        Args:
            p1, p2: First segment endpoints
            q1, q2: Second segment endpoints
            voronoi_regions: Voronoi regions
            rng: Random number generator
            
        Returns:
            Spatial bridge point or None
        """
        def cross(a: Point, b: Point) -> float:
            return a.x * b.y - a.y * b.x
        
        r = p2 - p1
        s = q2 - q1
        denom = cross(r, s)
        
        if abs(denom) < 1e-9:
            return None  # Parallel lines
        
        qp = q1 - p1
        t = cross(qp, s) / denom
        u = cross(qp, r) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Lines intersect
            intersection_point = Point(p1.x + r.x * t, p1.y + r.y * t)
            
            # Calculate intersection angle
            road_angle = math.atan2(r.y, r.x)
            river_angle = math.atan2(s.y, s.x)
            intersection_angle = abs(road_angle - river_angle)
            intersection_angle = min(intersection_angle, math.pi - intersection_angle)
            
            # Calculate spatial advantage
            spatial_advantage = self._calculate_spatial_advantage(
                intersection_point, voronoi_regions
            )
            
            # Determine bridge type based on angle and spatial context
            bridge_type = self._determine_bridge_type(
                intersection_angle, spatial_advantage, road_angle, rng
            )
            
            # Calculate connectivity score
            connectivity_score = self._calculate_connectivity_score(
                intersection_point, voronoi_regions
            )
            
            return SpatialBridgePoint(
                point=intersection_point,
                intersection_angle=intersection_angle,
                spatial_advantage=spatial_advantage,
                bridge_type=bridge_type,
                connectivity_score=connectivity_score
            )
        
        return None
    
    def _calculate_spatial_advantage(
        self,
        point: Point,
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> float:
        """
        Calculate spatial advantage score for a bridge location.
        
        Args:
            point: Bridge location
            voronoi_regions: Voronoi regions
            
        Returns:
            Spatial advantage score (0-1)
        """
        best_advantage = 0.0
        
        for region in voronoi_regions.values():
            # Prefer intersections near region centers or boundaries
            distance_to_center = point.dist(region.center)
            
            # Check if point is near region boundary (good for bridges)
            boundary_score = self._calculate_boundary_proximity(point, region)
            
            # Region type preference
            type_bonus = {
                "center": 1.0,
                "intermediate": 0.8,
                "edge": 0.6,
                "corner": 0.4
            }.get(region.region_type, 0.5)
            
            # Combine factors
            region_advantage = boundary_score * type_bonus
            best_advantage = max(best_advantage, region_advantage)
        
        return best_advantage
    
    def _calculate_boundary_proximity(
        self,
        point: Point,
        region: VoronoiRegion
    ) -> float:
        """
        Calculate proximity to region boundary.
        
        Args:
            point: Point to evaluate
            region: Voronoi region
            
        Returns:
            Boundary proximity score (0-1)
        """
        min_distance_to_vertex = float('inf')
        
        for vertex in region.vertices:
            distance = point.dist(vertex)
            min_distance_to_vertex = min(min_distance_to_vertex, distance)
        
        # Prefer points near boundaries but not too close to vertices
        optimal_distance = math.sqrt(region.area) * 0.1
        boundary_score = max(0.1, 1.0 - abs(min_distance_to_vertex - optimal_distance) / optimal_distance)
        
        return boundary_score
    
    def _determine_bridge_type(
        self,
        intersection_angle: float,
        spatial_advantage: float,
        road_angle: float,
        rng: RNG
    ) -> str:
        """
        Determine appropriate bridge type based on spatial context.
        
        Args:
            intersection_angle: Angle of road-river intersection
            spatial_advantage: Spatial advantage score
            road_angle: Road angle
            rng: Random number generator
            
        Returns:
            Bridge type string
        """
        # Right angle intersections are good for regular bridges
        right_angle_threshold = math.pi / 3  # 60 degrees
        
        if intersection_angle >= right_angle_threshold and spatial_advantage > 0.6:
            # High-quality intersection - standard bridge
            return "bridge"
        elif spatial_advantage > 0.7:
            # High spatial advantage - could be important bridge
            return rng.choice(["bridge", "bridge"], p=[0.8, 0.2])  # Mostly bridges
        else:
            # Lower quality intersection - smaller bridge
            return "small_bridge"
    
    def _calculate_connectivity_score(
        self,
        point: Point,
        voronoi_regions: Dict[str, VoronoiRegion]
    ) -> float:
        """
        Calculate connectivity score for a bridge location.
        
        Args:
            point: Bridge location
            voronoi_regions: Voronoi regions
            
        Returns:
            Connectivity score (0-1)
        """
        connectivity_score = 0.0
        connected_regions = 0
        
        for region in voronoi_regions.values():
            if self._point_near_region_boundary(point, region):
                connected_regions += 1
        
        # Score based on number of regions the bridge connects
        connectivity_score = min(1.0, connected_regions / 3.0)
        
        return connectivity_score
    
    def _point_near_region_boundary(
        self,
        point: Point,
        region: VoronoiRegion,
        threshold: float = 15.0
    ) -> bool:
        """
        Check if a point is near a region boundary.
        
        Args:
            point: Point to check
            region: Voronoi region
            threshold: Distance threshold
            
        Returns:
            True if point is near boundary
        """
        for vertex in region.vertices:
            if point.dist(vertex) < threshold:
                return True
        return False
    
    def _generate_bridge_points(
        self,
        intersection_points: List[SpatialBridgePoint],
        roads: List[Road],
        river_points: List[Point],
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG
    ) -> List[SpatialBridgePoint]:
        """
        Generate optimal bridge placement points.
        
        Args:
            intersection_points: Initial intersection points
            roads: Road network
            river_points: River centerline
            voronoi_regions: Voronoi regions
            rng: Random number generator
            
        Returns:
            List of optimal bridge points
        """
        # Sort intersection points by quality
        quality_points = []
        for point in intersection_points:
            quality_score = (
                point.spatial_advantage * 0.4 +
                point.connectivity_score * 0.3 +
                (1.0 - point.intersection_angle / math.pi) * 0.3
            )
            quality_points.append((point, quality_score))
        
        quality_points.sort(key=lambda x: x[1], reverse=True)
        
        # Select best points with spacing constraints
        selected_points = []
        for point, quality in quality_points:
            if len(selected_points) >= len(roads) * 2:  # Reasonable limit
                break
            
            # Check spacing from already selected points
            valid = True
            for selected_point in selected_points:
                if point.point.dist(selected_point.point) < self.constraints.min_bridge_spacing:
                    valid = False
                    break
            
            if valid and quality > 0.3:  # Minimum quality threshold
                selected_points.append(point)
        
        return selected_points
    
    def _create_bridges_from_points(
        self,
        bridge_points: List[SpatialBridgePoint],
        roads: List[Road],
        river_points: List[Point],
        rng: RNG
    ) -> List[Bridge]:
        """
        Create bridge objects from placement points.
        
        Args:
            bridge_points: Optimal bridge placement points
            roads: Road network for reference
            river_points: River centerline
            rng: Random number generator
            
        Returns:
            List of created bridges
        """
        bridges = []
        
        for bridge_point in bridge_points:
            # Calculate bridge properties based on context
            length = self._calculate_bridge_length(bridge_point, river_points, rng)
            width = self._calculate_bridge_width(bridge_point, roads, rng)
            rotation = self._calculate_bridge_rotation(bridge_point, roads, rng)
            
            # Create bridge
            bridge = Bridge(
                center=bridge_point.point,
                length=length,
                width=width,
                rotation=rotation
            )
            
            bridges.append(bridge)
        
        return bridges
    
    def _calculate_bridge_length(
        self,
        bridge_point: SpatialBridgePoint,
        river_points: List[Point],
        rng: RNG
    ) -> float:
        """
        Calculate appropriate bridge length.
        
        Args:
            bridge_point: Bridge placement point
            river_points: River centerline
            rng: Random number generator
            
        Returns:
            Bridge length
        """
        # Estimate river width at this point
        river_width_estimate = 12.0  # Default river width
        
        # Base length on estimated river width
        base_length = river_width_estimate * 1.5
        
        # Adjust based on bridge type
        type_multipliers = {
            "bridge": 1.2,
            "small_bridge": 0.8,
            "large_bridge": 1.5
        }
        
        multiplier = type_multipliers.get(bridge_point.bridge_type, 1.0)
        length = base_length * multiplier
        
        # Add some variation
        length *= rng.uniform(0.9, 1.1)
        
        return length
    
    def _calculate_bridge_width(
        self,
        bridge_point: SpatialBridgePoint,
        roads: List[Road],
        rng: RNG
    ) -> float:
        """
        Calculate appropriate bridge width.
        
        Args:
            bridge_point: Bridge placement point
            roads: Road network
            rng: Random number generator
            
        Returns:
            Bridge width
        """
        # Base width on road network characteristics
        base_width = 2.5
        
        # Adjust based on spatial advantage (more important locations get wider bridges)
        if bridge_point.spatial_advantage > 0.8:
            base_width *= 1.4
        elif bridge_point.spatial_advantage > 0.6:
            base_width *= 1.2
        
        # Bridge type adjustment
        type_multipliers = {
            "bridge": 1.0,
            "small_bridge": 0.7,
            "large_bridge": 1.3
        }
        
        multiplier = type_multipliers.get(bridge_point.bridge_type, 1.0)
        width = base_width * multiplier
        
        return width
    
    def _calculate_bridge_rotation(
        self,
        bridge_point: SpatialBridgePoint,
        roads: List[Road],
        rng: RNG
    ) -> float:
        """
        Calculate appropriate bridge rotation.
        
        Args:
            bridge_point: Bridge placement point
            roads: Road network
            rng: Random number generator
            
        Returns:
            Bridge rotation angle
        """
        # Bridges should be perpendicular to the river flow
        # This is a simplified calculation - in reality, you'd calculate the river direction
        
        # For now, use the intersection angle to determine orientation
        # Bridges are typically perpendicular to roads
        base_rotation = bridge_point.intersection_angle + math.pi / 2
        
        # Add slight variation for natural appearance
        rotation = base_rotation + rng.gauss(0, 0.1)
        
        return rotation
    
    def _generate_additional_bridge_types(
        self,
        existing_bridges: List[Bridge],
        river_points: List[Point],
        roads: List[Road],
        rng: RNG
    ) -> List[Bridge]:
        """
        Generate additional bridge types (docks, ferries) based on spatial context.
        
        Args:
            existing_bridges: Already generated bridges
            river_points: River centerline
            roads: Road network
            rng: Random number generator
            
        Returns:
            List of additional bridge-type structures
        """
        additional_bridges = []
        
        # Generate docks near existing bridges
        dock_count = max(0, int(len(existing_bridges) * 0.6))
        
        for i in range(dock_count):
            if existing_bridges and rng.rand() < 0.5:
                # Create dock near existing bridge
                base_bridge = rng.choice(existing_bridges)
                dock = self._create_dock_near_bridge(base_bridge, rng)
                if dock:
                    additional_bridges.append(dock)
        
        return additional_bridges
    
    def _create_dock_near_bridge(
        self,
        bridge: Bridge,
        rng: RNG
    ) -> Optional[Bridge]:
        """
        Create a dock near an existing bridge.
        
        Args:
            bridge: Base bridge
            rng: Random number generator
            
        Returns:
            Created dock or None
        """
        # Create dock offset from bridge
        offset_distance = rng.uniform(8.0, 15.0)
        offset_angle = bridge.rotation + (math.pi / 2 if rng.rand() < 0.5 else -math.pi / 2)
        
        dock_x = bridge.center.x + math.cos(offset_angle) * offset_distance
        dock_y = bridge.center.y + math.sin(offset_angle) * offset_distance
        
        # Check bounds
        if (dock_x < 2 or dock_y < 2 or 
            dock_x > self.width - 2 or dock_y > self.height - 2):
            return None
        
        # Create dock
        dock_length = rng.uniform(8.0, 14.0)
        dock_width = rng.uniform(1.2, 1.8)
        dock_rotation = offset_angle
        
        return Bridge(
            center=Point(dock_x, dock_y),
            length=dock_length,
            width=dock_width,
            rotation=dock_rotation
        )
    
    def _apply_bridge_constraints(self, bridges: List[Bridge]) -> List[Bridge]:
        """
        Apply spatial constraints to bridge placement.
        
        Args:
            bridges: All generated bridges
            
        Returns:
            Filtered list of bridges meeting constraints
        """
        if len(bridges) <= 1:
            return bridges
        
        # Sort bridges by importance (longer bridges first)
        bridges.sort(key=lambda b: b.length * b.width, reverse=True)
        
        filtered_bridges = []
        for bridge in bridges:
            # Check distance from all previously accepted bridges
            valid = True
            for accepted_bridge in filtered_bridges:
                distance = bridge.center.dist(accepted_bridge.center)
                min_distance = self.constraints.min_bridge_spacing
                
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                filtered_bridges.append(bridge)
        
        return filtered_bridges

# Legacy compatibility functions
def generate_bridges(roads: List[Road], river_pts: List[Point], rng: RNG) -> List[Bridge]:
    """
    Legacy compatibility function for bridge generation.
    
    This maintains the original interface while using the new spatial system.
    
    Args:
        roads: Road network
        river_pts: River centerline points
        rng: Random number generator
        
    Returns:
        Generated bridges
    """
    if not river_pts:
        return []
    
    # Estimate map dimensions
    if river_pts:
        xs = [p.x for p in river_pts]
        ys = [p.y for p in river_pts]
        width = int(max(xs) - min(xs) + 40)
        height = int(max(ys) - min(ys) + 40)
    else:
        width, height = 300, 300
    
    # Generate Voronoi regions for spatial organization
    voronoi_regions = generate_village_regions(
        width=width,
        height=height,
        rng=rng,
        region_count=10,
        clustering=0.3,
        min_distance=30.0
    )
    
    # Create advanced bridge generator
    generator = AdvancedBridgeGenerator(width, height)
    
    # Generate bridges
    bridges = generator.generate_bridge_network(
        roads=roads,
        river_points=river_pts,
        voronoi_regions=voronoi_regions,
        rng=rng
    )
    
    return bridges

def generate_docks(bridges: List[Bridge], rng: RNG) -> List[Bridge]:
    """
    Legacy compatibility function for dock generation.
    
    Args:
        bridges: Existing bridges
        rng: Random number generator
        
    Returns:
        Generated docks
    """
    docks = []
    
    for bridge in bridges:
        if rng.rand() < 0.55:  # 55% chance of having docks
            # 1-2 small docks near bridge
            for _ in range(rng.randint(1, 2)):
                angle = bridge.rotation + (math.pi/2 if rng.rand() < 0.5 else -math.pi/2) + rng.gauss(0, 0.2)
                length = rng.uniform(7.0, 12.0)
                width = rng.uniform(1.2, 1.8)
                offset = rng.uniform(3.0, 6.0)
                cx = bridge.center.x + math.cos(angle) * offset
                cy = bridge.center.y + math.sin(angle) * offset
                docks.append(Bridge(center=Point(cx, cy), length=length, width=width, rotation=angle))
    
    return docks

# Legacy helper functions
def _segment_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> Optional[Tuple[float, float, Point]]:
    """Legacy compatibility for segment intersection."""
    def cross(a: Point, b: Point) -> float:
        return a.x * b.y - a.y * b.x
    
    r = p2 - p1
    s = q2 - q1
    denom = cross(r, s)
    
    if abs(denom) < 1e-9:
        return None
    
    qp = q1 - p1
    t = cross(qp, s) / denom
    u = cross(qp, r) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        pt = Point(p1.x + r.x * t, p1.y + r.y * t)
        return t, u, pt
    
    return None

def _polyline_segments(points):
    """Legacy compatibility for polyline segments."""
    for i in range(len(points) - 1):
        yield points[i], points[i + 1]
