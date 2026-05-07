"""
madrs — Modified Acceleration-Displacement Response Spectrum method.

Quickly run a seismic performance-point analysis:

    from madrs import MADRS_Method, curve_intersections, closest_point_on_curve
"""

from .core import (
    MADRS_Method,
    area_between_curves,
    curve_intersections,
    closest_point_on_curve,
)

__all__ = [
    "MADRS_Method",
    "area_between_curves",
    "curve_intersections",
    "closest_point_on_curve",
]

__version__ = "0.1.0"
