"""
Legacy Noise System - Replaced by Simplex Noise Implementation

This module provides legacy compatibility for the original noise generation system
that has been replaced by the new Simplex noise implementation. The original
value noise and fractal noise functions are maintained for backward compatibility
but are now superseded by the more advanced Simplex noise system.

LEGACY COMPATIBILITY NOTICE:
============================
This module is maintained for legacy compatibility only. The village generation
system now uses Simplex noise (see simplex_noise.py) for superior terrain and
density generation. New implementations should use the Simplex noise system
directly for better performance and natural patterns.

For new development, use:
- generate_simplex_noise() from simplex_noise.py
- Multi-scale noise generation with Voronoi integration
- Improved noise patterns with reduced directional artifacts

KEY IMPROVEMENTS IN NEW SYSTEM:
- Simplex noise provides faster computation and better visual quality
- Multi-dimensional noise support (1D, 2D, 3D, 4D)
- Reduced directional artifacts compared to value noise
- Better clustering of features for natural terrain
- Native support for Voronoi diagram integration
- Enhanced parameter control for fine-tuning
"""

from __future__ import annotations
import numpy as np
from .rng import RNG

def value_noise(width: int, height: int, rng: RNG, grid: int = 8) -> np.ndarray:
    """
    Legacy value noise generation - replaced by Simplex noise.
    
    DEPRECATED: This function is maintained for legacy compatibility only.
    Use Simplex noise from simplex_noise.py for new implementations.
    
    The original value noise implementation used random grid points with
    bilinear interpolation. While functional, it produced directional artifacts
    and was computationally less efficient than Simplex noise.
    
    Args:
        width: Width of the output array
        height: Height of the output array
        rng: Random number generator
        grid: Grid size for noise generation
        
    Returns:
        2D array of noise values in range [0, 1]
    """
    # random grid then bilinear upsample
    gx = width // grid + 2
    gy = height // grid + 2
    base = np.array([[rng.rand() for _ in range(gx)] for __ in range(gy)], dtype=np.float32)
    
    # coordinates in base grid space
    ys = np.linspace(0, gy-1, height)
    xs = np.linspace(0, gx-1, width)
    xi = np.floor(xs).astype(int); xf = xs - xi
    yi = np.floor(ys).astype(int); yf = ys - yi
    
    out = np.zeros((height, width), dtype=np.float32)
    
    # Bilinear interpolation with corner blending
    for j in range(height):
        y0 = yi[j]; y1 = min(y0+1, gy-1); ty = yf[j]
        row0 = base[y0]; row1 = base[y1]
        
        # Interpolate horizontally then vertically
        v0 = row0[xi]*(1-xf) + row0[np.minimum(xi+1, gx-1)]*xf
        v1 = row1[xi]*(1-xf) + row1[np.minimum(xi+1, gx-1)]*xf
        out[j] = v0*(1-ty) + v1*ty
    
    return out

def fractal_noise(width: int, height: int, rng: RNG, octaves: int = 4) -> np.ndarray:
    """
    Legacy fractal noise generation - replaced by Simplex noise.
    
    DEPRECATED: This function is maintained for legacy compatibility only.
    Use Simplex noise fractal generation for new implementations.
    
    The original fractal noise combined multiple octaves of value noise with
    decreasing amplitude. While effective, it suffered from the same limitations
    as value noise and was less efficient than multi-scale Simplex noise.
    
    This implementation created terrain-like patterns by:
    1. Generating noise at different frequencies (grid sizes)
    2. Combining octaves with decreasing amplitude
    3. Using a simple frequency doubling scheme
    
    Args:
        width: Width of the output array
        height: Height of the output array
        rng: Random number generator
        octaves: Number of noise layers to combine
        
    Returns:
        2D array of fractal noise values in range [0, 1]
    """
    acc = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    total = 0.0
    
    for o in range(octaves):
        # Double the frequency each octave (traditional approach)
        grid = max(2, 2**(o+2))
        n = value_noise(width, height, rng, grid=grid)
        
        # Accumulate with decreasing amplitude (0.5 factor)
        acc += n * amp
        total += amp
        amp *= 0.5
    
    # Normalize to [0, 1] range
    return acc / total if total > 0 else acc

# MIGRATION GUIDE:
# ================
"""
To migrate from legacy noise to Simplex noise:

OLD (Legacy):
-------------
from .noise import fractal_noise
density = fractal_noise(width, height, rng, octaves=4)

NEW (Simplex):
--------------
from .simplex_noise import generate_simplex_noise
density = generate_simplex_noise(
    width=width,
    height=height,
    rng=rng,
    scale=1.0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0
)

BENEFITS OF NEW SYSTEM:
----------------------
1. Better visual quality with reduced artifacts
2. Faster computation for complex patterns
3. More natural terrain generation
4. Better integration with Voronoi diagrams
5. Enhanced parameter control
6. Multi-dimensional noise support
7. Improved clustering of features

For advanced usage, see the SimplexNoise class in simplex_noise.py
for full control over noise generation parameters and multi-scale patterns.
"""
