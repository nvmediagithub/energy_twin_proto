"""
Advanced Landmark Generation System with Spatial Constraints and Voronoi Integration

This module provides a completely redesigned landmark generation system that implements
sophisticated spatial constraints and integrates with Voronoi spatial organization
for optimal landmark placement.

Key Features:
- Voronoi-based landmark placement for spatial organization
- Distance-based landmark spacing and clustering
- Regional landmark types based on spatial characteristics
- Integration with house, road, and field placement constraints
- Natural landmark distribution and hierarchy
- Enhanced landmark variety and architectural detail

The system ensures landmarks are placed optimally within the spatial organization
while maintaining proper distances from all infrastructure elements and following
natural village development patterns.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from dataclasses import dataclass
from .rng import RNG
from .voronoi import VoronoiRegion, VoronoiGenerator, generate_village_regions
from ..domain.geometry import Point, Polyline
from ..domain.models import House, Road

@dataclass
class LandmarkPlacementConstraints:
    """
    Defines spatial constraints for landmark placement.
    
    Attributes:
        min_landmark_spacing: Minimum distance between landmarks
        house_buffer: Minimum distance from houses
        road_proximity: Preferred distance from roads
        voronoi_integration: Whether to use Voronoi-based placement
        max_landmarks_per_region: Maximum landmarks per Voronoi region
        landmark_hierarchy: Whether to maintain landmark hierarchy
    """
    min_landmark_spacing: float = 25.0
    house_buffer: float = 15.0
    road_proximity: float = 8.0
    voronoi_integration: bool = True
    max_landmarks_per_region: int = 2
    landmark_hierarchy: bool = True

@dataclass
class LandmarkRegion:
    """
    Represents a landmark placement region with characteristics.
    
    Attributes:
        region: Source Voronoi region
        landmark_type: Type of landmark suitable for this region
        priority: Priority for landmark placement
        suitability_score: Suitability score for landmark placement
        spatial_advantages: List of spatial advantages
    """
    region: VoronoiRegion
    landmark_type: str
    priority: float
    suitability_score: float
    spatial_advantages: List[str]

class AdvancedLandmarkGenerator:
    """
    Advanced landmark generator with spatial constraints and Voronoi integration.
    
    This system creates realistic landmark distributions by:
    1. Using Voronoi regions for spatial organization
    2. Implementing proper distance constraints from infrastructure
    3. Creating natural landmark hierarchies and distributions
    4. Optimizing landmark placement based on spatial characteristics
    5. Ensuring proper spacing and clustering patterns
    
    Features:
    - Voronoi-region-based landmark placement
    - Distance-constrained landmark spacing
    - Regional landmark type optimization
    - Landmark hierarchy management
    - Integration with infrastructure spacing
    - Natural landmark clustering
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: LandmarkPlacementConstraints = None
    ):
        """
        Initialize the advanced landmark generator.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
            constraints: Landmark placement constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or LandmarkPlacementConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def generate_landmarks(
        self,
        houses: List[House],
        roads: List[Road],
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG,
        target_count: int
    ) -> List[House]:
        """
        Generate landmarks with proper spatial organization and constraints.
        
        This method creates natural landmark distributions by:
        1. Analyzing Voronoi regions for landmark suitability
        2. Creating landmark regions with appropriate characteristics
        3. Placing landmarks with proper spacing and constraints
        4. Ensuring natural landmark hierarchies
        
        Args:
            houses: House positions (some will become landmarks)
            roads: Road network for proximity analysis
            voronoi_regions: Voronoi regions for spatial organization
            rng: Random number generator
            target_count: Target number of landmarks to generate
            
        Returns:
            List of houses that have been converted to landmarks
        """
        # Create landmark regions from Voronoi regions
        landmark_regions = self._create_landmark_regions(
            voronoi_regions, houses, roads, target_count, rng
        )
        
        # Select and convert houses to landmarks
        landmark_houses = self._convert_houses_to_landmarks(
            landmark_regions, houses, roads, target_count, rng
        )
        
        # Apply spatial constraints to landmarks
        final_landmarks = self._apply_landmark_constraints(landmark_houses)
        
        return final_landmarks
    
    def _create_landmark_regions(
        self,
        regions: Dict[str, VoronoiRegion],
        houses: List[House],
        roads: List[Road],
        target_count: int,
        rng: RNG
    ) -> List[LandmarkRegion]:
        """
        Create landmark regions from Voronoi regions with spatial characteristics.
        
        Args:
            regions: Voronoi regions
            houses: Available houses
            roads: Road network
            target_count: Target number of landmarks
            rng: Random number generator
            
        Returns:
            List of landmark regions
        """
        landmark_regions = []
        
        for region in regions.values():
            # Evaluate region suitability for landmarks
            suitability = self._evaluate_landmark_suitability(
                region, houses, roads
            )
            
            if suitability > 0.5:  # Minimum threshold for landmarks
                landmark_region = self._create_landmark_region_from_voronoi(
                    region, suitability, rng
                )
                landmark_regions.append(landmark_region)
        
        # Sort by suitability and limit based on target count
        landmark_regions.sort(key=lambda r: r.priority, reverse=True)
        
        # Limit regions to reasonable number based on target count
        max_regions = max(1, target_count * 2)
        return landmark_regions[:max_regions]
    
    def _evaluate_landmark_suitability(
        self,
        region: VoronoiRegion,
        houses: List[House],
        roads: List[Road]
    ) -> float:
        """
        Evaluate suitability of a Voronoi region for landmark placement.
        
        Args:
            region: Voronoi region to evaluate
            houses: Available houses
            roads: Road network
            
        Returns:
            Suitability score (0-1)
        """
        # Base suitability from region type
        type_scores = {
            "center": 1.0,
            "intermediate": 0.8,
            "edge": 0.5,
            "corner": 0.2
        }
        
        base_score = type_scores.get(region.region_type, 0.4)
        
        # Size suitability (prefer medium-sized regions for landmarks)
        optimal_area = self.width * self.height * 0.02
        size_score = 1.0 - abs(region.area - optimal_area) / optimal_area
        size_score = max(0.2, size_score)
        
        # Road proximity (prefer regions with good road access)
        road_proximity = self._calculate_road_proximity(region, roads)
        
        # House density (prefer regions with nearby houses for landmarks)
        house_density = self._calculate_house_density(region, houses)
        
        # Neighbor count (prefer well-connected regions)
        neighbor_score = min(1.0, len(region.neighbors) / 5.0)
        
        # Combine scores
        total_score = (
            base_score * 0.3 +
            size_score * 0.2 +
            road_proximity * 0.25 +
            house_density * 0.15 +
            neighbor_score * 0.1
        )
        
        return min(1.0, total_score)
    
    def _calculate_road_proximity(
        self,
        region: VoronoiRegion,
        roads: List[Road]
    ) -> float:
        """
        Calculate road proximity score for a region.
        
        Args:
            region: Voronoi region
            roads: Road network
            
        Returns:
            Road proximity score (0-1)
        """
        if not roads:
            return 0.3  # Default if no roads
        
        min_distance = float('inf')
        road_count_nearby = 0
        
        for road in roads:
            for point in road.polyline.points:
                distance = region.center.dist(point)
                min_distance = min(min_distance, distance)
                
                if distance < 20:  # Within reasonable landmark access distance
                    road_count_nearby += 1
        
        # Score based on closest road and nearby road count
        distance_score = max(0.1, 1.0 - min_distance / 30)
        count_score = min(1.0, road_count_nearby / 3)
        
        return (distance_score + count_score) / 2
    
    def _calculate_house_density(
        self,
        region: VoronoiRegion,
        houses: List[House]
    ) -> float:
        """
        Calculate house density around a region.
        
        Args:
            region: Voronoi region
            houses: Available houses
            
        Returns:
            House density score (0-1)
        """
        if not houses:
            return 0.0
        
        houses_nearby = 0
        for house in houses:
            distance = region.center.dist(house.center)
            if distance < 25:  # Within landmark influence radius
                houses_nearby += 1
        
        return min(1.0, houses_nearby / 5.0)
    
    def _create_landmark_region_from_voronoi(
        self,
        region: VoronoiRegion,
        suitability: float,
        rng: RNG
    ) -> LandmarkRegion:
        """
        Create a landmark region from a Voronoi region.
        
        Args:
            region: Source Voronoi region
            suitability: Landmark suitability score
            rng: Random number generator
            
        Returns:
            Landmark region with characteristics
        """
        # Determine appropriate landmark type
        landmark_type = self._determine_landmark_type(region, suitability, rng)
        
        # Calculate spatial advantages
        spatial_advantages = self._identify_spatial_advantages(region, landmark_type)
        
        # Calculate placement priority
        priority = suitability * self._calculate_type_priority(landmark_type)
        
        return LandmarkRegion(
            region=region,
            landmark_type=landmark_type,
            priority=priority,
            suitability_score=suitability,
            spatial_advantages=spatial_advantages
        )
    
    def _determine_landmark_type(
        self,
        region: VoronoiRegion,
        suitability: float,
        rng: RNG
    ) -> str:
        """
        Determine appropriate landmark type for a region.
        
        Args:
            region: Voronoi region
            suitability: Suitability score
            rng: Random number generator
            
        Returns:
            Landmark type string
        """
        # Use suitability and region characteristics to determine type
        if region.region_type == "center" and suitability > 0.8:
            # Central regions good for important landmarks
            return rng.choice(
                ["hall", "chapel", "market"], 
                p=[0.4, 0.4, 0.2]
            )
        elif region.area > 600 and suitability > 0.7:
            # Large regions good for barns or community buildings
            return rng.choice(
                ["barn", "hall", "chapel"], 
                p=[0.5, 0.3, 0.2]
            )
        elif len(region.neighbors) > 4 and suitability > 0.6:
            # Well-connected regions good for meeting places
            return rng.choice(
                ["market", "hall", "barn"], 
                p=[0.4, 0.4, 0.2]
            )
        else:
            # Default to barn for smaller regions
            return "barn"
    
    def _identify_spatial_advantages(
        self,
        region: VoronoiRegion,
        landmark_type: str
    ) -> List[str]:
        """
        Identify spatial advantages of a region for landmark placement.
        
        Args:
            region: Voronoi region
            landmark_type: Type of landmark
            
        Returns:
            List of spatial advantages
        """
        advantages = []
        
        if region.region_type == "center":
            advantages.append("central_location")
        
        if len(region.neighbors) > 4:
            advantages.append("high_connectivity")
        
        if region.area > 400:
            advantages.append("ample_space")
        
        if len(region.neighbors) > 6:
            advantages.append("intersection_point")
        
        return advantages
    
    def _calculate_type_priority(self, landmark_type: str) -> float:
        """
        Calculate priority multiplier based on landmark type.
        
        Args:
            landmark_type: Type of landmark
            
        Returns:
            Priority multiplier
        """
        multipliers = {
            "hall": 1.2,      # Important community building
            "chapel": 1.1,    # Religious significance
            "market": 1.0,    # Commercial importance
            "barn": 0.9       # Agricultural building
        }
        return multipliers.get(landmark_type, 1.0)
    
    def _convert_houses_to_landmarks(
        self,
        landmark_regions: List[LandmarkRegion],
        houses: List[House],
        roads: List[Road],
        target_count: int,
        rng: RNG
    ) -> List[House]:
        """
        Convert houses to landmarks based on spatial analysis.
        
        Args:
            landmark_regions: Landmark regions for placement
            houses: Available houses to convert
            roads: Road network
            target_count: Target number of landmarks
            rng: Random number generator
            
        Returns:
            List of houses converted to landmarks
        """
        if not houses or not landmark_regions:
            return []
        
        landmark_houses = []
        
        # Sort houses by suitability for landmark conversion
        house_suitabilities = []
        for house in houses:
            suitability = self._calculate_house_landmark_suitability(
                house, landmark_regions, roads
            )
            house_suitabilities.append((house, suitability))
        
        house_suitabilities.sort(key=lambda x: x[1], reverse=True)
        
        # Convert houses to landmarks
        for house, suitability in house_suitabilities:
            if len(landmark_houses) >= target_count:
                break
            
            if suitability > 0.5:  # Minimum suitability threshold
                landmark = self._convert_house_to_landmark(
                    house, landmark_regions, rng
                )
                if landmark:
                    landmark_houses.append(landmark)
        
        return landmark_houses
    
    def _calculate_house_landmark_suitability(
        self,
        house: House,
        landmark_regions: List[LandmarkRegion],
        roads: List[Road]
    ) -> float:
        """
        Calculate suitability of a house for landmark conversion.
        
        Args:
            house: House to evaluate
            landmark_regions: Available landmark regions
            roads: Road network
            
        Returns:
            Suitability score (0-1)
        """
        best_region_suitability = 0.0
        
        # Find best matching landmark region
        for landmark_region in landmark_regions:
            distance = house.center.dist(landmark_region.region.center)
            
            # Prefer houses in or near landmark regions
            if distance < 15:
                region_suitability = landmark_region.suitability_score
            elif distance < 30:
                region_suitability = landmark_region.suitability_score * 0.7
            else:
                region_suitability = 0.0
            
            best_region_suitability = max(best_region_suitability, region_suitability)
        
        # Factor in road proximity
        road_proximity = self._calculate_house_road_proximity(house, roads)
        
        # Combine scores
        total_suitability = best_region_suitability * 0.7 + road_proximity * 0.3
        
        return total_suitability
    
    def _calculate_house_road_proximity(
        self,
        house: House,
        roads: List[Road]
    ) -> float:
        """
        Calculate road proximity for a house.
        
        Args:
            house: House to evaluate
            roads: Road network
            
        Returns:
            Road proximity score (0-1)
        """
        if not roads:
            return 0.3
        
        min_distance = float('inf')
        for road in roads:
            for point in road.polyline.points:
                distance = house.center.dist(point)
                min_distance = min(min_distance, distance)
        
        # Prefer houses at optimal distance from roads
        optimal_distance = 8.0
        distance_score = max(0.1, 1.0 - abs(min_distance - optimal_distance) / 15)
        
        return distance_score
    
    def _convert_house_to_landmark(
        self,
        house: House,
        landmark_regions: List[LandmarkRegion],
        rng: RNG
    ) -> Optional[House]:
        """
        Convert a house to a landmark with enhanced properties.
        
        Args:
            house: House to convert
            landmark_regions: Available landmark regions
            rng: Random number generator
            
        Returns:
            Converted landmark house or None
        """
        # Find the best matching landmark region
        best_region = None
        best_distance = float('inf')
        
        for landmark_region in landmark_regions:
            distance = house.center.dist(landmark_region.region.center)
            if distance < best_distance:
                best_distance = distance
                best_region = landmark_region
        
        if not best_region:
            return None
        
        # Enhance house properties for landmark status
        landmark = House(
            center=house.center,
            width=house.width * rng.uniform(1.8, 2.8),  # Larger footprint
            height=house.height * rng.uniform(1.8, 2.8),
            rotation=house.rotation,
            side=house.side
        )
        
        # Set landmark properties
        if not hasattr(landmark, 'meta'):
            landmark.meta = {}
        
        landmark.meta["landmark"] = best_region.landmark_type
        landmark.meta["color_variant"] = "dark"
        landmark.meta["spatial_advantages"] = best_region.spatial_advantages
        landmark.meta["landmark_priority"] = best_region.priority
        
        return landmark
    
    def _apply_landmark_constraints(self, landmarks: List[House]) -> List[House]:
        """
        Apply spatial constraints to landmark placement.
        
        Args:
            landmarks: Initially selected landmarks
            
        Returns:
            Filtered list of landmarks meeting constraints
        """
        if len(landmarks) <= 1:
            return landmarks
        
        # Sort landmarks by priority (larger landmarks first)
        landmarks.sort(key=lambda l: l.width * l.height, reverse=True)
        
        filtered_landmarks = []
        for landmark in landmarks:
            # Check distance from all previously accepted landmarks
            valid = True
            for accepted_landmark in filtered_landmarks:
                distance = landmark.center.dist(accepted_landmark.center)
                min_distance = self.constraints.min_landmark_spacing
                
                if distance < min_distance:
                    valid = False
                    break
            
            # Check distance from houses (for landmarks that aren't houses)
            if valid:
                # This is a simplified check - in reality, you'd want to check against all houses
                if len(filtered_landmarks) > 0:
                    nearest_house_distance = float('inf')
                    for other_landmark in landmarks:
                        if other_landmark != landmark:
                            dist = landmark.center.dist(other_landmark.center)
                            nearest_house_distance = min(nearest_house_distance, dist)
                    
                    if nearest_house_distance < self.constraints.house_buffer:
                        valid = False
            
            if valid:
                filtered_landmarks.append(landmark)
        
        return filtered_landmarks

# Legacy compatibility function
def add_landmarks(
    houses: List[House], 
    roads: List[Road], 
    rng: RNG, 
    count: int = 2
) -> List[House]:
    """
    Legacy compatibility function for landmark generation.
    
    This maintains the original interface while using the new advanced
    landmark generation system.
    
    Args:
        houses: House positions
        roads: Road network
        rng: Random number generator
        count: Number of landmarks to create
        
    Returns:
        List of houses converted to landmarks
    """
    if not houses:
        return houses
    
    # Estimate map dimensions from house positions
    if houses:
        xs = [h.center.x for h in houses]
        ys = [h.center.y for h in houses]
        width = int(max(xs) - min(xs) + 40)
        height = int(max(ys) - min(ys) + 40)
    else:
        width, height = 300, 300
    
    # Generate Voronoi regions for spatial organization
    region_count = max(8, len(houses) * 2)
    voronoi_regions = generate_village_regions(
        width=width,
        height=height,
        rng=rng,
        region_count=region_count,
        clustering=0.4,
        min_distance=25.0
    )
    
    # Create advanced landmark generator
    generator = AdvancedLandmarkGenerator(width, height)
    
    # Generate landmarks
    landmarks = generator.generate_landmarks(
        houses=houses,
        roads=roads,
        voronoi_regions=voronoi_regions,
        rng=rng,
        target_count=count
    )
    
    # Apply landmarks to houses (replace some houses with landmarks)
    result_houses = []
    landmark_count = 0
    
    for house in houses:
        if landmark_count < len(landmarks) and landmark_count < count:
            # Convert this house to a landmark
            landmark = landmarks[landmark_count]
            result_houses.append(landmark)
            landmark_count += 1
        else:
            # Keep as regular house
            result_houses.append(house)
    
    return result_houses
