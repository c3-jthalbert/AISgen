# aisgen/__init__.py

"""
AISgen: Synthetic AIS track generation for maritime simulation.
Primary usage is in notebooks.
"""
from .environment import AOIEnvironment
from .vessel import VesselTemplate
from .tracks import TrackGenerator, TrackBuilder
from .utils import snap_polyline_to_poisson, random_poisson_in_polygon  # if present

__all__ = [
    "AOIEnvironment",
    "VesselTemplate",
    "TrackGenerator",
    "TrackBuilder",
    "snap_polyline_to_poisson",
    "random_poisson_in_polygon",
]

