"""
Advanced Field Generation System with Voronoi Region Integration

This module provides a completely redesigned field generation system that leverages
Voronoi regions for natural field placement and spatial organization. The new system
replaces the previous house-centered approach with region-based field distribution
for more realistic agricultural patterns.

Key Features:
- Voronoi-based field placement for natural agricultural regions
- Distance-based spacing between fields and infrastructure
- Regional field types (fields, orchards, specialty crops)
- Natural field shapes with irregular boundaries
- Integration with houses, roads, and terrain for optimal placement
- Field size optimization based on region characteristics

The system now uses Voronoi regions as the primary organizing principle for
agricultural areas, ensuring fields follow natural spatial boundaries and
maintain proper distances from all infrastructure elements.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from .rng import RNG
from .voronoi import VoronoiRegion, VoronoiGenerator
from ..domain.geometry import Point, Polyline
from ..domain.models import House, FieldPlot, Road

@dataclass
class FieldPlacementConstraints:
    """
    Defines spatial constraints for field placement.
    
    Attributes:
        min_field_spacing: Minimum distance between fields
        house_buffer: Minimum distance from houses
        road_buffer: Minimum distance from roads
        voronoi_integration: Whether to use Voronoi-based placement
        max_fields_per_region: Maximum fields per Voronoi region
        field_size_factor: Overall field size multiplier
    """
    min_field_spacing: float = 15.0
    house_buffer: float = 8.0
    road_buffer: float = 5.0
    voronoi_integration: bool = True
    max_fields_per_region: int = 3
    field_size_factor: float = 1.0

@dataclass
class FieldRegion:
    """
    Represents a field region with agricultural characteristics.
    
    Attributes:
        region: Source Voronoi region
        field_type: Type of agricultural field
        size_category: Size category (small, medium, large)
        placement_priority: Priority for field placement
        assigned_houses: Houses that can use this field
    """
    region: VoronoiRegion
    field_type: str
    size_category: str
    placement_priority: float
    assigned_houses: List[House]

class AdvancedFieldGenerator:
    """
    Advanced field generator using Voronoi spatial organization.
    
    This system creates realistic agricultural patterns by:
    1. Using Voronoi regions for spatial organization
    2. Implementing proper distance constraints from infrastructure
    3. Creating natural field shapes with irregular boundaries
    4. Optimizing field placement based on regional characteristics
    5. Ensuring proper spacing between agricultural areas
    
    Features:
    - Voronoi-region-based field placement
    - Distance-constrained field spacing
    - Natural field shape generation
    - Regional agricultural optimization
    - House-field assignment optimization
    - Integration with infrastructure spacing
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: FieldPlacementConstraints = None
    ):
        """
        Initialize the advanced field generator.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
            constraints: Field placement constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or FieldPlacementConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def generate_agricultural_regions(
        self,
        houses: List[House],
        voronoi_regions: Dict[str, VoronoiRegion],
        roads: List[Road],
        rng: RNG,
        map_w: int,
        map_h: int
    ) -> List[FieldPlot]:
        """
        Generate a complete agricultural system using Voronoi regions.
        
        This method creates natural agricultural patterns by:
        1. Analyzing Voronoi regions for agricultural suitability
        2. Creating field regions with appropriate characteristics
        3. Placing fields with proper spacing and constraints
        4. Generating natural field shapes and boundaries
        
        Args:
            houses: House positions for field assignment
            voronoi_regions: Voronoi regions for spatial organization
            roads: Road network for spacing constraints
            rng: Random number generator
            map_w, map_h: Map dimensions
            
        Returns:
            List of generated field plots
        """
        # Create field regions from Voronoi regions
        field_regions = self._create_field_regions(
            voronoi_regions, houses, roads, rng
        )
        
        # Assign houses to field regions
        self._assign_houses_to_regions(field_regions, houses)
        
        # Generate fields for each region
        all_fields = []
        
        for field_region in field_regions:
            region_fields = self._generate_fields_for_region(
                field_region, roads, rng
            )
            all_fields.extend(region_fields)
        
        # Apply global field spacing constraints
        optimized_fields = self._apply_field_spacing_constraints(all_fields)
        
        return optimized_fields
    
    def _create_field_regions(
        self,
        regions: Dict[str, VoronoiRegion],
        houses: List[House],
        roads: List[Road],
        rng: RNG
    ) -> List[FieldRegion]:
        """
        Create field regions from Voronoi regions with agricultural characteristics.
        
        Args:
            regions: Voronoi regions
            houses: Available houses
            roads: Road network
            rng: Random number generator
            
        Returns:
            List of field regions
        """
        field_regions = []
        
        for region in regions.values():
            # Evaluate region suitability for agriculture
            suitability = self._evaluate_agricultural_suitability(
                region, houses, roads
            )
            
            if suitability > 0.4:  # Minimum threshold for agriculture
                field_region = self._create_field_region_from_voronoi(
                    region, suitability, rng
                )
                field_regions.append(field_region)
        
        # Sort by suitability and limit number
        field_regions.sort(key=lambda f: f.placement_priority, reverse=True)
        
        return field_regions
    
    def _evaluate_agricultural_suitability(
        self,
        region: VoronoiRegion,
        houses: List[House],
        roads: List[Road]
    ) -> float:
        """
        Evaluate agricultural suitability of a Voronoi region.
        
        Args:
            region: Voronoi region to evaluate
            houses: Available houses
            roads: Road network
            
        Returns:
            Suitability score (0-1)
        """
        # Base suitability from region type
        type_scores = {
            "center": 0.9,
            "intermediate": 0.8,
            "edge": 0.6,
            "corner": 0.3
        }
        
        base_score = type_scores.get(region.region_type, 0.5)
        
        # Size suitability (prefer medium-sized regions)
        optimal_area = self.width * self.height * 0.015
        size_score = 1.0 - abs(region.area - optimal_area) / optimal_area
        size_score = max(0.1, size_score)
        
        # Proximity to houses (prefer regions with nearby houses)
        house_proximity = self._calculate_house_proximity(region, houses)
        
        # Distance from roads (prefer some road access)
        road_distance = self._calculate_min_road_distance(region, roads)
        road_score = min(1.0, max(0.2, 1.0 - road_distance / 50))
        
        # Combine scores
        total_score = (
            base_score * 0.3 +
            size_score * 0.25 +
            house_proximity * 0.25 +
            road_score * 0.2
        )
        
        return min(1.0, total_score)
    
    def _calculate_house_proximity(
        self,
        region: VoronoiRegion,
        houses: List[House]
    ) -> float:
        """
        Calculate proximity of houses to a region.
        
        Args:
            region: Voronoi region
            houses: Available houses
            
        Returns:
            Proximity score (0-1)
        """
        if not houses:
            return 0.0
        
        min_distance = float('inf')
        house_count_nearby = 0
        
        for house in houses:
            distance = region.center.dist(house.center)
            min_distance = min(min_distance, distance)
            
            if distance < 30:  # Within reasonable farming distance
                house_count_nearby += 1
        
        # Score based on closest house and nearby house count
        distance_score = max(0.1, 1.0 - min_distance / 60)
        count_score = min(1.0, house_count_nearby / 3)
        
        return (distance_score + count_score) / 2
    
    def _calculate_min_road_distance(
        self,
        region: VoronoiRegion,
        roads: List[Road]
    ) -> float:
        """
        Calculate minimum distance from region to any road.
        
        Args:
            region: Voronoi region
            roads: Road network
            
        Returns:
            Minimum distance to roads
        """
        if not roads:
            return 50.0  # Default distance if no roads
        
        min_distance = float('inf')
        
        for road in roads:
            for point in road.polyline.points:
                distance = region.center.dist(point)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _create_field_region_from_voronoi(
        self,
        region: VoronoiRegion,
        suitability: float,
        rng: RNG
    ) -> FieldRegion:
        """
        Create a field region from a Voronoi region.
        
        Args:
            region: Source Voronoi region
            suitability: Agricultural suitability score
            rng: Random number generator
            
        Returns:
            Field region with agricultural characteristics
        """
        # Determine field type based on region characteristics
        field_type = self._determine_field_type(region, suitability, rng)
        
        # Determine size category
        size_category = self._determine_field_size_category(region)
        
        # Calculate placement priority
        priority = suitability * self._calculate_size_priority(size_category)
        
        return FieldRegion(
            region=region,
            field_type=field_type,
            size_category=size_category,
            placement_priority=priority,
            assigned_houses=[]
        )
    
    def _determine_field_type(
        self,
        region: VoronoiRegion,
        suitability: float,
        rng: RNG
    ) -> str:
        """
        Determine appropriate field type for a region.
        
        Args:
            region: Voronoi region
            suitability: Agricultural suitability
            rng: Random number generator
            
        Returns:
            Field type string
        """
        # Use suitability to influence field type choice
        r = rng.rand()
        
        if region.region_type == "center" and suitability > 0.7:
            # Central regions good for diverse crops
            if r < 0.6:
                return "field"
            elif r < 0.9:
                return "orchard"
            else:
                return "specialty"
        elif region.area > 800 and suitability > 0.6:
            # Large regions good for fields
            if r < 0.8:
                return "field"
            else:
                return "orchard"
        elif region.region_type == "edge" and suitability > 0.5:
            # Edge regions good for orchards
            if r < 0.7:
                return "orchard"
            else:
                return "field"
        else:
            # Default to field
            return "field"
    
    def _determine_field_size_category(self, region: VoronoiRegion) -> str:
        """
        Determine field size category based on region characteristics.
        
        Args:
            region: Voronoi region
            
        Returns:
            Size category string
        """
        optimal_area = self.width * self.height * 0.015
        
        if region.area < optimal_area * 0.5:
            return "small"
        elif region.area > optimal_area * 2:
            return "large"
        else:
            return "medium"
    
    def _calculate_size_priority(self, size_category: str) -> float:
        """
        Calculate priority multiplier based on size category.
        
        Args:
            size_category: Size category
            
        Returns:
            Priority multiplier
        """
        multipliers = {
            "small": 0.8,
            "medium": 1.0,
            "large": 1.2
        }
        return multipliers.get(size_category, 1.0)
    
    def _assign_houses_to_regions(
        self,
        field_regions: List[FieldRegion],
        houses: List[House]
    ) -> None:
        """
        Assign houses to field regions based on proximity.
        
        Args:
            field_regions: Available field regions
            houses: Houses to assign
        """
        for house in houses:
            # Find closest suitable region
            best_region = None
            best_distance = float('inf')
            
            for field_region in field_regions:
                distance = house.center.dist(field_region.region.center)
                
                # Consider region capacity
                capacity_factor = 1.0 - len(field_region.assigned_houses) / self.constraints.max_fields_per_region
                
                if capacity_factor > 0 and distance < best_distance:
                    best_distance = distance
                    best_region = field_region
            
            # Assign house to the best region if reasonable distance
            if best_region and best_distance < 40:
                best_region.assigned_houses.append(house)
    
    def _generate_fields_for_region(
        self,
        field_region: FieldRegion,
        roads: List[Road],
        rng: RNG
    ) -> List[FieldPlot]:
        """
        Generate fields for a single field region.
        
        Args:
            field_region: Field region for field generation
            roads: Road network for constraints
            rng: Random number generator
            
        Returns:
            List of fields generated for this region
        """
        fields = []
        
        # Determine number of fields based on region characteristics
        max_fields = self.constraints.max_fields_per_region
        base_field_count = self._calculate_base_field_count(field_region)
        field_count = min(max_fields, base_field_count)
        
        # Generate fields
        for i in range(field_count):
            field = self._generate_single_field(
                field_region, roads, i, field_count, rng
            )
            if field:
                fields.append(field)
        
        return fields
    
    def _calculate_base_field_count(self, field_region: FieldRegion) -> int:
        """
        Calculate base number of fields for a region.
        
        Args:
            field_region: Field region
            
        Returns:
            Base field count
        """
        optimal_area = self.width * self.height * 0.015
        
        # Base count on region area
        area_factor = field_region.region.area / optimal_area
        
        # Adjust based on field type
        if field_region.field_type == "orchard":
            # Orchards tend to be smaller but more numerous
            count = max(1, int(area_factor * 0.8))
        elif field_region.field_type == "specialty":
            # Specialty crops are fewer but larger
            count = max(1, int(area_factor * 0.6))
        else:
            # Regular fields
            count = max(1, int(area_factor))
        
        return count
    
    def _generate_single_field(
        self,
        field_region: FieldRegion,
        roads: List[Road],
        field_index: int,
        total_fields: int,
        rng: RNG
    ) -> Optional[FieldPlot]:
        """
        Generate a single field within a field region.
        
        Args:
            field_region: Field region
            roads: Road network
            field_index: Index of this field
            total_fields: Total fields in region
            rng: Random number generator
            
        Returns:
            Generated field or None if generation fails
        """
        # Calculate field characteristics
        center = self._calculate_field_center(field_region, field_index, total_fields, rng)
        size = self._calculate_field_size(field_region, rng)
        shape = self._generate_field_shape(center, size, field_region.field_type, rng)
        
        # Validate field placement
        if not self._validate_field_placement(shape, roads, field_region):
            return None
        
        # Create field plot
        field = FieldPlot(polygon=shape, kind=field_region.field_type)
        return field
    
    def _calculate_field_center(
        self,
        field_region: FieldRegion,
        field_index: int,
        total_fields: int,
        rng: RNG
    ) -> Point:
        """
        Calculate optimal center position for a field.
        
        Args:
            field_region: Field region
            field_index: Index of the field
            total_fields: Total fields in region
            rng: Random number generator
            
        Returns:
            Field center point
        """
        base_center = field_region.region.center
        
        if total_fields == 1:
            # Single field - center in region
            offset_x = rng.gauss(0, field_region.region.area * 0.01)
            offset_y = rng.gauss(0, field_region.region.area * 0.01)
            return Point(base_center.x + offset_x, base_center.y + offset_y)
        
        else:
            # Multiple fields - distribute around region center
            angle = (2 * math.pi * field_index) / total_fields + rng.gauss(0, 0.3)
            distance = math.sqrt(field_region.region.area) * 0.3
            
            x = base_center.x + math.cos(angle) * distance
            y = base_center.y + math.sin(angle) * distance
            
            return Point(x, y)
    
    def _calculate_field_size(
        self,
        field_region: FieldRegion,
        rng: RNG
    ) -> Tuple[float, float]:
        """
        Calculate field dimensions based on region characteristics.
        
        Args:
            field_region: Field region
            rng: Random number generator
            
        Returns:
            Tuple of (width, height)
        """
        # Base size from region area and field type
        base_area = field_region.region.area * 0.6  # Use 60% of region for fields
        
        if field_region.field_type == "orchard":
            # Orchards are typically smaller
            base_area *= 0.7
        elif field_region.field_type == "specialty":
            # Specialty crops can be larger
            base_area *= 1.3
        
        # Calculate dimensions with aspect ratio
        aspect_ratio = rng.uniform(0.7, 1.4)
        
        if aspect_ratio >= 1:
            width = math.sqrt(base_area * aspect_ratio)
            height = width / aspect_ratio
        else:
            height = math.sqrt(base_area / aspect_ratio)
            width = height * aspect_ratio
        
        # Apply size constraints
        width = max(8, min(width, 60))
        height = max(6, min(height, 50))
        
        # Apply global size factor
        width *= self.constraints.field_size_factor
        height *= self.constraints.field_size_factor
        
        return width, height
    
    def _generate_field_shape(
        self,
        center: Point,
        size: Tuple[float, float],
        field_type: str,
        rng: RNG
    ) -> List[Point]:
        """
        Generate natural field shape with irregular boundaries.
        
        Args:
            center: Field center point
            size: Field dimensions (width, height)
            field_type: Type of field
            rng: Random number generator
            
        Returns:
            List of points forming the field boundary
        """
        width, height = size
        
        # Start with rectangle
        vertices = [
            Point(center.x - width/2, center.y - height/2),
            Point(center.x + width/2, center.y - height/2),
            Point(center.x + width/2, center.y + height/2),
            Point(center.x - width/2, center.y + height/2)
        ]
        
        # Add irregularity based on field type
        if field_type == "orchard":
            # Orchards have more irregular shapes
            vertices = self._add_shape_irregularity(vertices, 0.15, rng)
        elif field_type == "specialty":
            # Specialty crops have moderate irregularity
            vertices = self._add_shape_irregularity(vertices, 0.1, rng)
        else:
            # Regular fields have slight irregularity
            vertices = self._add_shape_irregularity(vertices, 0.08, rng)
        
        # Smooth corners
        vertices = self._smooth_field_corners(vertices, rng)
        
        return vertices
    
    def _add_shape_irregularity(
        self,
        vertices: List[Point],
        irregularity_factor: float,
        rng: RNG
    ) -> List[Point]:
        """
        Add natural irregularity to field shape.
        
        Args:
            vertices: Base vertices
            irregularity_factor: Amount of irregularity
            rng: Random number generator
            
        Returns:
            Vertices with added irregularity
        """
        irregular_vertices = []
        
        for i, vertex in enumerate(vertices):
            # Add random offset
            offset_x = rng.gauss(0, irregularity_factor * 2)
            offset_y = rng.gauss(0, irregularity_factor * 2)
            
            irregular_vertex = Point(
                vertex.x + offset_x,
                vertex.y + offset_y
            )
            irregular_vertices.append(irregular_vertex)
        
        return irregular_vertices
    
    def _smooth_field_corners(
        self,
        vertices: List[Point],
        rng: RNG
    ) -> List[Point]:
        """
        Apply corner smoothing for natural appearance.
        
        Args:
            vertices: Vertices to smooth
            rng: Random number generator
            
        Returns:
            Smoothed vertices
        """
        smoothed = []
        n = len(vertices)
        
        for i in range(n):
            prev_vertex = vertices[(i - 1) % n]
            curr_vertex = vertices[i]
            next_vertex = vertices[(i + 1) % n]
            
            # Calculate corner point with smoothing
            corner_x = (curr_vertex.x + prev_vertex.x + next_vertex.x) / 3
            corner_y = (curr_vertex.y + prev_vertex.y + next_vertex.y) / 3
            
            # Add slight smoothing noise
            corner_x += rng.gauss(0, 0.3)
            corner_y += rng.gauss(0, 0.3)
            
            smoothed.append(Point(corner_x, corner_y))
        
        return smoothed
    
    def _validate_field_placement(
        self,
        vertices: List[Point],
        roads: List[Road],
        field_region: FieldRegion
    ) -> bool:
        """
        Validate field placement against constraints.
        
        Args:
            vertices: Field vertices
            roads: Road network
            field_region: Field region
            
        Returns:
            True if placement is valid
        """
        # Check bounds
        for vertex in vertices:
            if (vertex.x < 2 or vertex.y < 2 or 
                vertex.x > self.width - 2 or vertex.y > self.height - 2):
                return False
        
        # Check distance from roads
        for road in roads:
            for point in road.polyline.points:
                for vertex in vertices:
                    if point.dist(vertex) < self.constraints.road_buffer:
                        return False
        
        # Check if field is within its region
        if not self._field_within_region(vertices, field_region.region):
            return False
        
        return True
    
    def _field_within_region(
        self,
        vertices: List[Point],
        region: VoronoiRegion
    ) -> bool:
        """
        Check if all field vertices are within the target region.
        
        Args:
            vertices: Field vertices
            region: Target Voronoi region
            
        Returns:
            True if all vertices are within the region
        """
        # Simple check: most vertices should be within reasonable distance of region center
        center = region.center
        max_distance = math.sqrt(region.area) * 0.7
        
        vertices_within = 0
        for vertex in vertices:
            if vertex.dist(center) <= max_distance:
                vertices_within += 1
        
        return vertices_within >= len(vertices) * 0.75  # 75% of vertices must be within
    
    def _apply_field_spacing_constraints(self, fields: List[FieldPlot]) -> List[FieldPlot]:
        """
        Apply global spacing constraints between fields.
        
        Args:
            fields: All generated fields
            
        Returns:
            Filtered list of fields meeting spacing constraints
        """
        if len(fields) <= 1:
            return fields
        
        # Sort fields by area (larger fields first)
        fields.sort(key=lambda f: self._calculate_field_area(f), reverse=True)
        
        filtered_fields = []
        for field in fields:
            # Check distance from all previously accepted fields
            valid = True
            for accepted_field in filtered_fields:
                if self._fields_too_close(field, accepted_field):
                    valid = False
                    break
            
            if valid:
                filtered_fields.append(field)
        
        return filtered_fields
    
    def _calculate_field_area(self, field: FieldPlot) -> float:
        """
        Calculate area of a field plot.
        
        Args:
            field: Field plot
            
        Returns:
            Field area
        """
        if len(field.polygon) < 3:
            return 0.0
        
        area = 0.0
        n = len(field.polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += field.polygon[i].x * field.polygon[j].y
            area += field.polygon[j].x * field.polygon[i].y
        
        return abs(area) / 2.0
    
    def _fields_too_close(self, field1: FieldPlot, field2: FieldPlot) -> bool:
        """
        Check if two fields are too close together.
        
        Args:
            field1: First field
            field2: Second field
            
        Returns:
            True if fields are too close
        """
        # Check minimum distance between any vertices
        for p1 in field1.polygon:
            for p2 in field2.polygon:
                if p1.dist(p2) < self.constraints.min_field_spacing:
                    return True
        
        return False

# Legacy compatibility function
def generate_fields(houses: List[House], rng: RNG, map_w: int, map_h: int) -> List[FieldPlot]:
    """
    Legacy compatibility function for field generation.
    
    This maintains the original interface while using the new Voronoi-based system.
    
    Args:
        houses: House positions
        rng: Random number generator
        map_w, map_h: Map dimensions
        
    Returns:
        Generated field plots
    """
    # Generate Voronoi regions for spatial organization
    region_count = max(8, len(houses) * 2)
    voronoi_regions = generate_village_regions(
        width=map_w,
        height=map_h,
        rng=rng,
        region_count=region_count,
        clustering=0.3,
        min_distance=20.0
    )
    
    # Create advanced field generator
    generator = AdvancedFieldGenerator(map_w, map_h)
    
    # Note: We need roads for proper spacing, but legacy function doesn't provide them
    # For now, we'll create an empty road list
    empty_roads = []
    
    # Generate fields
    fields = generator.generate_agricultural_regions(
        houses=houses,
        voronoi_regions=voronoi_regions,
        roads=empty_roads,
        rng=rng,
        map_w=map_w,
        map_h=map_h
    )
    
    return fields

# Legacy helper functions
def _irregular_quad(cx, cy, w, h, rng: RNG) -> List[Point]:
    """
    Legacy compatibility function for irregular quadrilateral generation.
    
    Args:
        cx, cy: Center coordinates
        w, h: Width and height
        rng: Random number generator
        
    Returns:
        List of points forming irregular quadrilateral
    """
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
    ]
    out = []
    for x, y in pts:
        out.append(Point(x + rng.gauss(0, w * 0.08), y + rng.gauss(0, h * 0.08)))
    return out
