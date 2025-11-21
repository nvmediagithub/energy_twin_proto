"""
Simplex Noise Implementation for Terrain Generation

This module provides a complete implementation of Simplex noise, which offers
significant improvements over traditional value noise for terrain generation.

Simplex noise advantages:
- Faster computation than Perlin noise for higher dimensions
- Reduced directional artifacts
- Better clustering of features
- More natural terrain patterns
- Reduced computational complexity for multi-octave generation

Based on Ken Perlin's improved noise algorithm and Stefan Gustavson's
Simplex noise implementation.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from .rng import RNG

class SimplexNoise:
    """
    Simplex noise generator for multi-dimensional noise generation.
    
    Features:
    - Multi-dimensional noise generation (1D, 2D, 3D, 4D)
    - Configurable octaves for fractal noise
    - Deterministic generation based on seed
    - Optimized for performance and memory usage
    - Natural terrain-like patterns
    """
    
    # Gradient tables for different dimensions
    _GRAD_1D = np.array([
        [1], [-1]
    ], dtype=np.float32)
    
    _GRAD_2D = np.array([
        [1, 1], [-1, 1], [1, -1], [-1, -1],
        [1, 0], [-1, 0], [0, 1], [0, -1]
    ], dtype=np.float32)
    
    _GRAD_3D = np.array([
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]
    ], dtype=np.float32)
    
    # Permutation table for random shuffling
    _DEFAULT_PERM = np.array([
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
        140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
        247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
        57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
        74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
        60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
        65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
        200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3,
        64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85,
        212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
        213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43,
        172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185,
        112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191,
        179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
        181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150,
        254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195,
        78, 66, 215, 61, 156, 180
    ], dtype=np.int32)
    
    def __init__(self, seed: int = 0):
        """
        Initialize Simplex noise generator with a seed.
        
        Args:
            seed: Random seed for deterministic generation
        """
        self._perm = self._generate_permutation(seed)
        
    def _generate_permutation(self, seed: int) -> np.ndarray:
        """
        Generate a seeded permutation table for the noise generator.
        
        Args:
            seed: Random seed for shuffling
            
        Returns:
            Shuffled permutation array
        """
        # Create a simple linear congruential generator for shuffling
        lcg_state = seed
        
        def lcg() -> int:
            nonlocal lcg_state
            lcg_state = (lcg_state * 1664525 + 1013904223) % (2**32)
            return lcg_state
        
        # Start with default permutation and shuffle it
        perm = self._DEFAULT_PERM.copy()
        n = len(perm)
        
        for i in range(n - 1, 0, -1):
            j = lcg() % (i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        
        # Duplicate the permutation for efficiency
        return np.tile(perm, 2)
    
    def noise_1d(self, x: float) -> float:
        """
        Generate 1D simplex noise.
        
        Args:
            x: Input coordinate
            
        Returns:
            Noise value in range [-1, 1]
        """
        # Calculate corner points
        xi = int(np.floor(x)) & 255
        xf = x - np.floor(x)
        
        # Calculate contribution from each corner
        t = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        
        aa = self._perm[xi]
        ab = self._perm[xi + 1]
        
        # Calculate gradient contributions
        x1 = xf
        x2 = xf - 1
        
        n0 = x1 * self._GRAD_1D[aa & 1][0]
        n1 = x2 * self._GRAD_1D[ab & 1][0]
        
        # Interpolate between corners
        return t * (n1 - n0) + n0
    
    def noise_2d(self, x: float, y: float) -> float:
        """
        Generate 2D simplex noise with improved directional distribution.
        
        The 2D simplex noise algorithm uses a triangular grid that provides
        better distribution and fewer directional artifacts than Perlin noise.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Noise value in range [-1, 1]
        """
        # Skew factor for triangular grid
        F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        G2 = (3.0 - np.sqrt(3.0)) / 6.0
        
        # Skew the input space
        s = (x + y) * F2
        i = int(np.floor(x + s))
        j = int(np.floor(y + s))
        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0
        
        # Determine which simplex we are in
        i1 = 1 if x0 > y0 else 0
        j1 = 1 if x0 <= y0 else 0
        
        # Offsets for corners
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2
        
        # Get permutation indices
        ii = i & 255
        jj = j & 255
        gi0 = self._perm[ii + self._perm[jj]] % 8
        gi1 = self._perm[ii + i1 + self._perm[jj + j1]] % 8
        gi2 = self._perm[ii + 1 + self._perm[jj + 1]] % 8
        
        # Calculate noise contributions
        n0 = self._corner_contribution_2d(gi0, x0, y0)
        n1 = self._corner_contribution_2d(gi1, x1, y1)
        n2 = self._corner_contribution_2d(gi2, x2, y2)
        
        # Combine contributions with fade curve
        return 70.0 * (n0 + n1 + n2)
    
    def _corner_contribution_2d(self, gi: int, x: float, y: float) -> float:
        """
        Calculate noise contribution from a corner point in 2D.
        
        Args:
            gi: Gradient index
            x: X coordinate relative to corner
            y: Y coordinate relative to corner
            
        Returns:
            Noise contribution value
        """
        grad = self._GRAD_2D[gi]
        t = 0.5 - x*x - y*y
        if t < 0:
            return 0.0
        t *= t
        return t * t * (grad[0] * x + grad[1] * y)
    
    def noise_3d(self, x: float, y: float, z: float) -> float:
        """
        Generate 3D simplex noise.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            
        Returns:
            Noise value in range [-1, 1]
        """
        # Skew factors for 3D simplex grid
        F3 = 1.0/3.0
        G3 = 1.0/6.0
        
        # Skew the input space
        s = (x + y + z) * F3
        i = int(np.floor(x + s))
        j = int(np.floor(y + s))
        k = int(np.floor(z + s))
        t = (i + j + k) * G3
        X0 = i - t
        Y0 = j - t
        Z0 = k - t
        x0 = x - X0
        y0 = y - Y0
        z0 = z - Z0
        
        # Determine simplex corner ordering
        i1, j1, k1 = self._simplex_order_3d(x0, y0, z0)
        
        # Offsets for second corner
        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        
        # Offsets for third corner
        x2 = x0 - i1 - j1 + 2.0 * G3
        y2 = y0 - j1 - k1 + 2.0 * G3
        z2 = z0 - k1 - i1 + 2.0 * G3
        
        # Offsets for fourth corner
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3
        
        # Get permutation indices
        ii = i & 255
        jj = j & 255
        kk = k & 255
        gi0 = self._perm[ii + self._perm[jj + self._perm[kk]]] % 12
        gi1 = self._perm[ii + i1 + self._perm[jj + j1 + self._perm[kk + k1]]] % 12
        gi2 = self._perm[ii + 1 + self._perm[jj + 1 + self._perm[kk + 1]]] % 12
        
        # Calculate contributions
        n0 = self._corner_contribution_3d(gi0, x0, y0, z0)
        n1 = self._corner_contribution_3d(gi1, x1, y1, z1)
        n2 = self._corner_contribution_3d(gi2, x2, y2, z2)
        n3 = self._corner_contribution_3d(gi2, x3, y3, z3)  # Note: gi2 for fourth corner
        
        return 32.0 * (n0 + n1 + n2 + n3)
    
    def _simplex_order_3d(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """
        Determine the ordering of corners for 3D simplex noise.
        
        Args:
            x, y, z: Coordinates
            
        Returns:
            Tuple of ordering indices
        """
        # Sort coordinates to determine corner ordering
        if x > y:
            if y > z:
                return 1, 1, 0
            elif x > z:
                return 1, 0, 1
            else:
                return 1, 0, 1
        else:
            if y < z:
                return 0, 0, 1
            elif x < z:
                return 0, 1, 0
            else:
                return 0, 1, 0
    
    def _corner_contribution_3d(self, gi: int, x: float, y: float, z: float) -> float:
        """
        Calculate noise contribution from a corner point in 3D.
        
        Args:
            gi: Gradient index
            x, y, z: Coordinates relative to corner
            
        Returns:
            Noise contribution value
        """
        grad = self._GRAD_3D[gi]
        t = 0.6 - x*x - y*y - z*z
        if t < 0:
            return 0.0
        t *= t
        return t * t * (grad[0] * x + grad[1] * y + grad[2] * z)
    
    def fractal_noise_2d(
        self, 
        width: int, 
        height: int, 
        scale: float = 1.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> np.ndarray:
        """
        Generate 2D fractal noise using multiple octaves of simplex noise.
        
        This method creates natural terrain-like patterns by combining
        multiple frequencies of simplex noise with decreasing amplitude.
        
        Args:
            width: Width of the output array
            height: Height of the output array
            scale: Scale factor for coordinates (higher = larger features)
            octaves: Number of noise layers to combine
            persistence: Amplitude decrease per octave (0.5 = half amplitude)
            lacunarity: Frequency increase per octave (2.0 = double frequency)
            offset_x: X offset for pattern variation
            offset_y: Y offset for pattern variation
            
        Returns:
            2D array of noise values normalized to [0, 1]
        """
        noise_array = np.zeros((height, width), dtype=np.float32)
        
        for octave in range(octaves):
            # Calculate frequency and amplitude for this octave
            frequency = lacunarity ** octave
            amplitude = persistence ** octave
            
            # Generate noise for this octave
            octave_noise = np.zeros((height, width), dtype=np.float32)
            
            for y in range(height):
                for x in range(width):
                    # Scale coordinates
                    nx = (x + offset_x) * frequency * scale / width
                    ny = (y + offset_y) * frequency * scale / height
                    
                    # Add simplex noise contribution
                    octave_noise[y, x] = self.noise_2d(nx, ny) * amplitude
            
            # Accumulate this octave
            noise_array += octave_noise
        
        # Normalize to [0, 1] range
        min_val = np.min(noise_array)
        max_val = np.max(noise_array)
        
        if max_val > min_val:
            noise_array = (noise_array - min_val) / (max_val - min_val)
        
        return noise_array
    
    def fractal_noise_3d(
        self, 
        width: int, 
        height: int, 
        depth: int,
        scale: float = 1.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0
    ) -> np.ndarray:
        """
        Generate 3D fractal noise using multiple octaves of simplex noise.
        
        Args:
            width, height, depth: Dimensions of the output array
            scale: Scale factor for coordinates
            octaves: Number of noise layers to combine
            persistence: Amplitude decrease per octave
            lacunarity: Frequency increase per octave
            offset_x, offset_y, offset_z: Coordinate offsets
            
        Returns:
            3D array of noise values normalized to [0, 1]
        """
        noise_array = np.zeros((height, width, depth), dtype=np.float32)
        
        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave
            
            octave_noise = np.zeros((height, width, depth), dtype=np.float32)
            
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        nx = (x + offset_x) * frequency * scale / width
                        ny = (y + offset_y) * frequency * scale / height
                        nz = (z + offset_z) * frequency * scale / depth
                        
                        octave_noise[y, x, z] = self.noise_3d(nx, ny, nz) * amplitude
            
            noise_array += octave_noise
        
        # Normalize to [0, 1] range
        min_val = np.min(noise_array)
        max_val = np.max(noise_array)
        
        if max_val > min_val:
            noise_array = (noise_array - min_val) / (max_val - min_val)
        
        return noise_array


def generate_simplex_noise(
    width: int, 
    height: int, 
    rng: RNG,
    scale: float = 1.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0
) -> np.ndarray:
    """
    Generate fractal simplex noise for terrain generation.
    
    This function provides a high-level interface for generating natural-looking
    terrain noise using the simplex noise algorithm.
    
    Args:
        width: Width of the output array
        height: Height of the output array
        rng: Random number generator for seed
        scale: Scale factor for feature size (higher = larger features)
        octaves: Number of detail layers to combine
        persistence: How much each octave contributes (0.5 = half amplitude)
        lacunarity: How much frequency increases per octave (2.0 = double frequency)
        
    Returns:
        2D array of noise values in range [0, 1] suitable for terrain generation
    """
    # Create simplex noise generator with seeded randomness
    simplex = SimplexNoise(seed=rng.seed)
    
    # Generate fractal noise with random offsets for variety
    offset_x = rng.uniform(0, 1000)
    offset_y = rng.uniform(0, 1000)
    
    noise = simplex.fractal_noise_2d(
        width=width,
        height=height,
        scale=scale,
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        offset_x=offset_x,
        offset_y=offset_y
    )
    
    return noise


def generate_multi_scale_noise(
    width: int, 
    height: int, 
    rng: RNG,
    scales: List[float] = None
) -> np.ndarray:
    """
    Generate multi-scale noise by combining different frequency bands.
    
    This function creates a more complex noise pattern by combining
    multiple scales of noise generation.
    
    Args:
        width: Width of the output array
        height: Height of the output array
        rng: Random number generator
        scales: List of scale factors to combine
        
    Returns:
        Combined noise array with multiple frequency bands
    """
    if scales is None:
        scales = [0.5, 1.0, 2.0, 4.0]
    
    combined_noise = np.zeros((height, width), dtype=np.float32)
    total_weight = 0.0
    
    for i, scale in enumerate(scales):
        # Calculate weight based on scale
        weight = 1.0 / (2.0 ** i)
        
        # Generate noise for this scale
        noise = generate_simplex_noise(
            width=width,
            height=height,
            rng=rng,
            scale=scale,
            octaves=3,
            persistence=0.7,
            lacunarity=2.0
        )
        
        # Combine with weight
        combined_noise += noise * weight
        total_weight += weight
    
    # Normalize
    if total_weight > 0:
        combined_noise /= total_weight
    
    return combined_noise