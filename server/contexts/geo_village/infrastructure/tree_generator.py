"""
Advanced Tree Generation System with Distance-Based Spacing and Voronoi Organization

This module provides a completely redesigned tree generation system that implements
sophisticated distance-based spacing algorithms and integrates with Voronoi spatial
organization for natural forest patterns.

Key Features:
- Distance-constrained tree placement for natural spacing
- Voronoi-based spatial organization for regional tree patterns
- Multi-scale forest clustering (forest blobs, orchards, scattered trees)
- Tree density optimization based on terrain and spatial constraints
- Natural forest edge and clearing generation
- Integration with houses, roads, and fields for proper spacing

The system ensures trees are placed with proper spacing from all infrastructure
elements while maintaining natural forest patterns and regional characteristics.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
import math
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from .rng import RNG
from .voronoi import VoronoiRegion, VoronoiGenerator
from ..domain.geometry import Point, Polyline
from ..domain.models import Tree, Road, House

@dataclass
class TreePlacementConstraints:
    """
    Defines spatial constraints for tree placement.
    
    Attributes:
        min_tree_spacing: Minimum distance between trees
        forest_cluster_spacing: Spacing within forest clusters
        orchard_spacing: Spacing between orchard trees
        house_buffer: Minimum distance from houses
        road_buffer: Minimum distance from roads
        voronoi_integration: Whether to use Voronoi-based placement
    """
    min_tree_spacing: float = 3.0
    forest_cluster_spacing: float = 4.0
    orchard_spacing: float = 2.5
    house_buffer: float = 6.0
    road_buffer: float = 3.0
    voronoi_integration: bool = True

@dataclass
class ForestRegion:
    """
    Represents a forest region with characteristics for tree placement.
    
    Attributes:
        center: Center point of the forest region
        radius: Radius of the forest region
        density: Tree density factor (0-1)
        forest_type: Type of forest (dense, sparse, mixed)
        tree_positions: Positions of trees in this region
    """
    center: Point
    radius: float
    density: float
    forest_type: str
    tree_positions: List[Point]

class AdvancedTreeGenerator:
    """
    Advanced tree generator with distance constraints and Voronoi organization.
    
    This system creates natural tree distributions by:
    1. Using Voronoi regions for spatial organization
    2. Implementing sophisticated distance constraints
    3. Creating natural forest clustering patterns
    4. Maintaining proper spacing from infrastructure
    5. Optimizing tree density based on regional characteristics
    
    Features:
    - Voronoi-region-based tree placement
    - Distance-constrained spacing algorithms
    - Multi-scale forest clustering
    - Natural forest edge generation
    - Integration with infrastructure spacing
    - Regional forest type optimization
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        constraints: TreePlacementConstraints = None
    ):
        """
        Initialize the advanced tree generator.
        
        Args:
            width: Width of the generation area
            height: Height of the generation area
            constraints: Tree placement constraints
        """
        self.width = width
        self.height = height
        self.constraints = constraints or TreePlacementConstraints()
        self.voronoi_generator = VoronoiGenerator(width, height)
        
    def generate_tree_distribution(
        self,
        density: np.ndarray,
        roads: List[Road],
        houses: List[House],
        voronoi_regions: Dict[str, VoronoiRegion],
        rng: RNG,
        target_count: int
    ) -> List[Tree]:
        """
        Generate a complete tree distribution with proper spacing and clustering.
        
        This method creates natural tree patterns by:
        1. Analyzing spatial constraints from infrastructure
        2. Creating Voronoi-based regional forest areas
        3. Placing trees with distance-based spacing
        4. Generating natural forest clusters and clearings
        
        Args:
            density: Density map for tree placement guidance
            roads: Road network for spacing constraints
            houses: House positions for spacing constraints
            voronoi_regions: Voronoi regions for spatial organization
            rng: Random number generator
            target_count: Target number of trees to generate
            
        Returns:
            List of placed trees with proper spacing
        """
        # Create spatial constraint maps
        constraint_map = self._create_constraint_map(roads, houses, voronoi_regions)
        
        # Generate forest regions for clustering
        forest_regions = self._generate_forest_regions(
            density, voronoi_regions, constraint_map, target_count, rng
        )
        
        # Place trees in forest regions
        all_trees = []
        
        for forest_region in forest_regions:
            region_trees = self._place_trees_in_forest_region(
                forest_region, constraint_map, rng
            )
            all_trees.extend(region_trees)
        
        # Add scattered individual trees
        scattered_trees = self._place_scattered_trees(
            density, constraint_map, target_count - len(all_trees), rng
        )
        all_trees.extend(scattered_trees)
        
        # Apply global spacing constraints
        final_trees = self._apply_global_tree_constraints(all_trees)
        
        return final_trees
    
    def _create_constraint_map(
        self,
        roads: List[Road],
        houses: List[House],
        regions: Dict[str, VoronoiRegion]
    ) -> np.ndarray:
        """
        Create a constraint map indicating where trees can and cannot be placed.
        
        Args:
            roads: Road network
            houses: House positions
            regions: Voronoi regions
            
        Returns:
            Constraint map (0 = no constraint, 1 = blocked)
        """
        constraint_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Add road constraints
        for road in roads:
            self._add_road_constraint(constraint_map, road)
        
        # Add house constraints
        for house in houses:
            self._add_house_constraint(constraint_map, house)
        
        # Add region-based constraints
        for region in regions.values():
            if region.region_type in ["corner"]:
                # Avoid placing trees in corner regions
                self._add_region_constraint(constraint_map, region, weight=0.8)
        
        return constraint_map
    
    def _add_road_constraint(self, constraint_map: np.ndarray, road: Road) -> None:
        """
        Add road spacing constraints to the constraint map.
        
        Args:
            constraint_map: Constraint map to modify
            road: Road to add constraints for
        """
        buffer_size = int(self.constraints.road_buffer)
        
        for point in road.polyline.points:
            ix, iy = int(point.x), int(point.y)
            
            x0 = max(0, ix - buffer_size)
            x1 = min(self.width, ix + buffer_size + 1)
            y0 = max(0, iy - buffer_size)
            y1 = min(self.height, iy + buffer_size + 1)
            
            # Validate ranges to prevent negative dimensions
            if x1 <= x0 or y1 <= y0:
                continue
            
            # Create distance-based constraint
            yy, xx = np.mgrid[y0:y1, x0:x1]
            distances = np.sqrt((xx - point.x) ** 2 + (yy - point.y) ** 2)
            
            # Apply constraint based on distance
            constraint = np.clip(1.0 - distances / buffer_size, 0, 1)
            constraint_map[y0:y1, x0:x1] = np.maximum(
                constraint_map[y0:y1, x0:x1], constraint * 0.7
            )
    
    def _add_house_constraint(self, constraint_map: np.ndarray, house: House) -> None:
        """
        Add house spacing constraints to the constraint map.
        
        Args:
            constraint_map: Constraint map to modify
            house: House to add constraints for
        """
        buffer_size = int(self.constraints.house_buffer)
        ix, iy = int(house.center.x), int(house.center.y)
        
        x0 = max(0, ix - buffer_size)
        x1 = min(self.width, ix + buffer_size + 1)
        y0 = max(0, iy - buffer_size)
        y1 = min(self.height, iy + buffer_size + 1)
        
        # Validate ranges to prevent negative dimensions
        if x1 <= x0 or y1 <= y0:
            return
        
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distances = np.sqrt((xx - house.center.x) ** 2 + (yy - house.center.y) ** 2)
        
        constraint = np.clip(1.0 - distances / buffer_size, 0, 1)
        constraint_map[y0:y1, x0:x1] = np.maximum(
            constraint_map[y0:y1, x0:x1], constraint
        )
    
    def _add_region_constraint(
        self,
        constraint_map: np.ndarray,
        region: VoronoiRegion,
        weight: float = 1.0
    ) -> None:
        """
        Add region-based constraints to the constraint map.
        
        Args:
            constraint_map: Constraint map to modify
            region: Voronoi region to add constraints for
            weight: Weight of the constraint
        """
        # Simple constraint based on region center and area
        center_x, center_y = int(region.center.x), int(region.center.y)
        radius = int(math.sqrt(region.area) * 0.3)
        
        x0 = max(0, center_x - radius)
        x1 = min(self.width, center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(self.height, center_y + radius + 1)
        
        # Validate ranges to prevent negative dimensions
        if x1 <= x0 or y1 <= y0:
            return
        
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        
        # Apply constraint with soft falloff
        constraint = np.exp(-distances / (radius * 0.5))
        constraint_map[y0:y1, x0:x1] = np.maximum(
            constraint_map[y0:y1, x0:x1], constraint * weight * 0.3
        )
    
    def _generate_forest_regions(
        self,
        density: np.ndarray,
        regions: Dict[str, VoronoiRegion],
        constraint_map: np.ndarray,
        target_count: int,
        rng: RNG
    ) -> List[ForestRegion]:
        """
        Generate forest regions for natural tree clustering.
        
        Args:
            density: Density map for tree placement
            regions: Voronoi regions for spatial organization
            constraint_map: Constraint map
            target_count: Target number of trees
            rng: Random number generator
            
        Returns:
            List of forest regions
        """
        forest_regions = []
        
        # Determine number of forest regions based on target count
        num_regions = max(1, min(target_count // 50, 8))
        
        # Find suitable regions for forest placement
        suitable_regions = self._find_suitable_forest_regions(regions, density)
        
        # Create forest regions
        for i in range(min(num_regions, len(suitable_regions))):
            region = suitable_regions[i % len(suitable_regions)]
            
            # Calculate forest characteristics
            radius = self._calculate_forest_radius(region, target_count, rng)
            density_factor = self._calculate_forest_density(region, density, constraint_map, rng)
            forest_type = self._determine_forest_type(region, density, rng)
            
            # Create forest region
            forest_region = ForestRegion(
                center=region.center,
                radius=radius,
                density=density_factor,
                forest_type=forest_type,
                tree_positions=[]
            )
            
            forest_regions.append(forest_region)
        
        return forest_regions
    
    def _find_suitable_forest_regions(
        self,
        regions: Dict[str, VoronoiRegion],
        density: np.ndarray
    ) -> List[VoronoiRegion]:
        """
        Find Voronoi regions suitable for forest placement.
        
        Args:
            regions: All Voronoi regions
            density: Density map
            
        Returns:
            List of suitable regions
        """
        suitable = []
        
        for region in regions.values():
            # Skip regions that are too small or in unsuitable locations
            if region.area < 100:
                continue
            
            if region.region_type == "center":
                # Central regions are good for small forest clusters
                if 200 <= region.area <= 800:
                    suitable.append(region)
            elif region.region_type == "intermediate":
                # Intermediate regions are good for larger forests
                if region.area > 300:
                    suitable.append(region)
            elif region.region_type == "edge":
                # Edge regions can have forest edges
                if region.area > 500:
                    suitable.append(region)
        
        # Sort by suitability
        suitable.sort(key=lambda r: r.area, reverse=True)
        
        return suitable
    
    def _calculate_forest_radius(
        self,
        region: VoronoiRegion,
        target_count: int,
        rng: RNG
    ) -> float:
        """
        Calculate appropriate radius for a forest region.
        
        Args:
            region: Voronoi region
            target_count: Target tree count
            rng: Random number generator
            
        Returns:
            Forest radius
        """
        # Base radius on region size
        base_radius = math.sqrt(region.area) * 0.3
        
        # Adjust based on region type
        if region.region_type == "center":
            radius_multiplier = rng.uniform(0.4, 0.7)
        elif region.region_type == "edge":
            radius_multiplier = rng.uniform(0.6, 1.0)
        else:
            radius_multiplier = rng.uniform(0.5, 0.8)
        
        return base_radius * radius_multiplier
    
    def _calculate_forest_density(
        self,
        region: VoronoiRegion,
        density: np.ndarray,
        constraint_map: np.ndarray,
        rng: RNG
    ) -> float:
        """
        Calculate appropriate tree density for a forest region.
        
        Args:
            region: Voronoi region
            density: Density map
            constraint_map: Constraint map
            rng: Random number generator
            
        Returns:
            Tree density factor (0-1)
        """
        # Get average density in region
        region_mask = self._create_region_mask(region)
        
        if not np.any(region_mask):
            return 0.5
        
        avg_density = np.mean(density[region_mask])
        avg_constraint = np.mean(constraint_map[region_mask])
        
        # Calculate density based on region characteristics
        base_density = 1.0 - avg_constraint  # Reduce density where there are constraints
        
        if region.region_type == "center":
            # Central regions can be denser
            density_factor = base_density * rng.uniform(0.7, 1.0)
        elif region.region_type == "edge":
            # Edge regions tend to be sparser
            density_factor = base_density * rng.uniform(0.4, 0.8)
        else:
            density_factor = base_density * rng.uniform(0.5, 0.9)
        
        return max(0.1, min(1.0, density_factor))
    
    def _determine_forest_type(
        self,
        region: VoronoiRegion,
        density: np.ndarray,
        rng: RNG
    ) -> str:
        """
        Determine the type of forest for a region.
        
        Args:
            region: Voronoi region
            density: Density map
            rng: Random number generator
            
        Returns:
            Forest type string
        """
        if region.region_type == "center" and region.area < 400:
            return "orchard"
        elif region.area > 1000:
            return "dense_forest"
        elif region.region_type == "edge":
            return "sparse_forest"
        else:
            # Simple choice without probabilities
            if rng.rand() < 0.6:
                return "mixed_forest"
            else:
                return "sparse_forest"
    
    def _create_region_mask(self, region: VoronoiRegion) -> np.ndarray:
        """
        Create a boolean mask for a Voronoi region.
        
        Args:
            region: Voronoi region
            
        Returns:
            Boolean mask for the region
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Simple circular mask approximation
        center_x, center_y = int(region.center.x), int(region.center.y)
        radius = int(math.sqrt(region.area) * 0.3)
        
        x0 = max(0, center_x - radius)
        x1 = min(self.width, center_x + radius)
        y0 = max(0, center_y - radius)
        y1 = min(self.height, center_y + radius)
        
        # Validate ranges to prevent negative dimensions
        if x1 <= x0 or y1 <= y0:
            return mask
        
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        mask[y0:y1, x0:x1] = distances <= radius
        
        return mask
    
    def _place_trees_in_forest_region(
        self,
        forest_region: ForestRegion,
        constraint_map: np.ndarray,
        rng: RNG
    ) -> List[Tree]:
        """
        Place trees within a forest region.
        
        Args:
            forest_region: Forest region for tree placement
            constraint_map: Constraint map
            rng: Random number generator
            
        Returns:
            List of trees placed in this region
        """
        trees = []
        
        # Determine placement strategy based on forest type
        if forest_region.forest_type == "orchard":
            trees = self._place_orchard_trees(forest_region, constraint_map, rng)
        elif forest_region.forest_type == "dense_forest":
            trees = self._place_dense_forest_trees(forest_region, constraint_map, rng)
        else:
            trees = self._place_sparse_forest_trees(forest_region, constraint_map, rng)
        
        # Store positions for the forest region
        forest_region.tree_positions = [tree.center for tree in trees]
        
        return trees
    
    def _place_orchard_trees(
        self,
        forest_region: ForestRegion,
        constraint_map: np.ndarray,
        rng: RNG
    ) -> List[Tree]:
        """
        Place trees in orchard pattern.
        
        Args:
            forest_region: Forest region
            constraint_map: Constraint map
            rng: Random number generator
            
        Returns:
            List of orchard trees
        """
        trees = []
        spacing = self.constraints.orchard_spacing
        
        # Create grid pattern for orchard
        grid_size = int(forest_region.radius * 2 / spacing)
        
        for i in range(-grid_size // 2, grid_size // 2):
            for j in range(-grid_size // 2, grid_size // 2):
                x = forest_region.center.x + i * spacing + rng.gauss(0, 0.3)
                y = forest_region.center.y + j * spacing + rng.gauss(0, 0.3)
                
                # Check if within forest region bounds
                distance_from_center = math.sqrt(
                    (x - forest_region.center.x) ** 2 + 
                    (y - forest_region.center.y) ** 2
                )
                
                if distance_from_center > forest_region.radius:
                    continue
                
                # Check constraints
                if not self._is_valid_tree_position(x, y, constraint_map, trees):
                    continue
                
                # Create tree
                tree = Tree(
                    center=Point(x, y),
                    radius=rng.uniform(0.8, 1.3)
                )
                trees.append(tree)
        
        return trees
    
    def _place_dense_forest_trees(
        self,
        forest_region: ForestRegion,
        constraint_map: np.ndarray,
        rng: RNG
    ) -> List[Tree]:
        """
        Place trees in dense forest pattern.
        
        Args:
            forest_region: Forest region
            constraint_map: Constraint map
            rng: Random number generator
            
        Returns:
            List of dense forest trees
        """
        trees = []
        target_count = int(forest_region.radius * 4 * forest_region.density)
        
        attempts = 0
        max_attempts = target_count * 3
        
        while len(trees) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Random point within forest region
            angle = rng.uniform(0, 2 * math.pi)
            distance = rng.uniform(0, forest_region.radius * 0.9)
            x = forest_region.center.x + math.cos(angle) * distance
            y = forest_region.center.y + math.sin(angle) * distance
            
            # Check constraints
            if not self._is_valid_tree_position(x, y, constraint_map, trees):
                continue
            
            # Create tree
            tree = Tree(
                center=Point(x, y),
                radius=rng.uniform(1.1, 2.2)
            )
            trees.append(tree)
        
        return trees
    
    def _place_sparse_forest_trees(
        self,
        forest_region: ForestRegion,
        constraint_map: np.ndarray,
        rng: RNG
    ) -> List[Tree]:
        """
        Place trees in sparse forest pattern.
        
        Args:
            forest_region: Forest region
            constraint_map: Constraint map
            rng: Random number generator
            
        Returns:
            List of sparse forest trees
        """
        trees = []
        target_count = int(forest_region.radius * 2 * forest_region.density)
        
        attempts = 0
        max_attempts = target_count * 5
        
        while len(trees) < target_count and attempts < max_attempts:
            attempts += 1
            
            # Random point with clustering tendency
            if rng.rand() < 0.3 and trees:
                # Place near existing tree
                parent_tree = rng.choice(trees)
                angle = rng.uniform(0, 2 * math.pi)
                distance = rng.uniform(
                    self.constraints.min_tree_spacing,
                    self.constraints.min_tree_spacing * 3
                )
                x = parent_tree.center.x + math.cos(angle) * distance
                y = parent_tree.center.y + math.sin(angle) * distance
            else:
                # Random placement within region
                angle = rng.uniform(0, 2 * math.pi)
                distance = rng.uniform(0, forest_region.radius * 0.8)
                x = forest_region.center.x + math.cos(angle) * distance
                y = forest_region.center.y + math.sin(angle) * distance
            
            # Check constraints
            if not self._is_valid_tree_position(x, y, constraint_map, trees):
                continue
            
            # Create tree
            tree = Tree(
                center=Point(x, y),
                radius=rng.uniform(0.7, 1.5)
            )
            trees.append(tree)
        
        return trees
    
    def _is_valid_tree_position(
        self,
        x: float,
        y: float,
        constraint_map: np.ndarray,
        existing_trees: List[Tree]
    ) -> bool:
        """
        Check if a tree position is valid.
        
        Args:
            x, y: Tree position
            constraint_map: Constraint map
            existing_trees: Already placed trees
            
        Returns:
            True if position is valid
        """
        # Check bounds
        if x < 1 or y < 1 or x >= self.width - 1 or y >= self.height - 1:
            return False
        
        # Check constraint map
        ix, iy = int(x), int(y)
        if constraint_map[iy, ix] > 0.6:
            return False
        
        # Check distance from existing trees
        for tree in existing_trees:
            distance = math.sqrt(
                (x - tree.center.x) ** 2 + (y - tree.center.y) ** 2
            )
            if distance < self.constraints.min_tree_spacing:
                return False
        
        return True
    
    def _place_scattered_trees(
        self,
        density: np.ndarray,
        constraint_map: np.ndarray,
        target_count: int,
        rng: RNG
    ) -> List[Tree]:
        """
        Place scattered individual trees outside forest regions.
        
        Args:
            density: Density map
            constraint_map: Constraint map
            target_count: Target number of scattered trees
            rng: Random number generator
            
        Returns:
            List of scattered trees
        """
        trees = []
        attempts = 0
        max_attempts = target_count * 10
        
        while len(trees) < target_count and attempts < max_attempts:
            attempts += 1
            
            x = rng.uniform(0, self.width - 1)
            y = rng.uniform(0, self.height - 1)
            
            # Check density preference (prefer lower density areas)
            ix, iy = int(x), int(y)
            if density[iy, ix] > 0.4:
                continue
            
            # Check constraints
            if not self._is_valid_tree_position(x, y, constraint_map, trees):
                continue
            
            # Create tree
            tree = Tree(
                center=Point(x, y),
                radius=rng.uniform(0.6, 1.2)
            )
            trees.append(tree)
        
        return trees
    
    def _apply_global_tree_constraints(self, trees: List[Tree]) -> List[Tree]:
        """
        Apply global spacing constraints to all trees.
        
        Args:
            trees: All placed trees
            
        Returns:
            Filtered list of trees meeting all constraints
        """
        if len(trees) <= 1:
            return trees
        
        # Sort trees by radius (larger trees first)
        trees.sort(key=lambda t: t.radius, reverse=True)
        
        filtered_trees = []
        for tree in trees:
            # Check distance from all previously accepted trees
            valid = True
            for accepted_tree in filtered_trees:
                distance = tree.center.dist(accepted_tree.center)
                min_distance = max(
                    self.constraints.min_tree_spacing,
                    (tree.radius + accepted_tree.radius) * 1.2
                )
                
                if distance < min_distance:
                    valid = False
                    break
            
            if valid:
                filtered_trees.append(tree)
        
        return filtered_trees

# Legacy compatibility function
def scatter_trees(
    density: np.ndarray,
    roads: List[Road],
    houses: List[House],
    rng: RNG,
    count: int,
    forest_blobs: int = 3
) -> List[Tree]:
    """
    Legacy compatibility function for tree scattering.
    
    This function maintains the original interface while using the new
    advanced tree generation system.
    
    Args:
        density: Density map for tree placement
        roads: Road network
        houses: House positions
        rng: Random number generator
        count: Target number of trees
        forest_blobs: Number of forest blobs (legacy parameter)
        
    Returns:
        List of placed trees with proper spacing
    """
    # Estimate map dimensions
    height, width = density.shape
    
    # Generate Voronoi regions for spatial organization
    region_count = max(8, count // 20)
    voronoi_regions = generate_village_regions(
        width=width,
        height=height,
        rng=rng,
        region_count=region_count,
        clustering=0.4,
        min_distance=30.0
    )
    
    # Create advanced tree generator
    generator = AdvancedTreeGenerator(width, height)
    
    # Generate tree distribution
    trees = generator.generate_tree_distribution(
        density=density,
        roads=roads,
        houses=houses,
        voronoi_regions=voronoi_regions,
        rng=rng,
        target_count=count
    )
    
    return trees
