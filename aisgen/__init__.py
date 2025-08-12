# aisgen/__init__.py

"""
AISgen: Synthetic AIS track generation for maritime simulation.

This package-level initializer re-exports the most commonly used classes and
helpers so users can do:
    from aisgen import AOIEnvironment, TrackGenerator, setup_aisgen_environment, ...
"""

__version__ = "0.2.0"

__all__ = []

# --- Core classes ---
from .environment import AOIEnvironment
__all__ += ["AOIEnvironment"]

from .vessel import VesselTemplate, VesselInstance, load_vessel_templates, sample_all
__all__ += ["VesselTemplate", "VesselInstance", "load_vessel_templates", "sample_all"]

from .tracks import TrackGenerator
__all__ += ["TrackGenerator"]

# --- High-level utilities ---
from .utils import setup_aisgen_environment, snap_polyline_to_poisson, random_poisson_in_polygon
__all__ += ["setup_aisgen_environment", "snap_polyline_to_poisson", "random_poisson_in_polygon"]

# --- GeoJSON helpers & plotting wrapper ---
from .geojson_utils import (
    load_geojson,
    polygon_to_feature_collection,
    gdf_to_feature_collection,
    plot_geojson_polygon,
)
__all__ += ["load_geojson", "polygon_to_feature_collection", "gdf_to_feature_collection", "plot_geojson_polygon"]

# --- Routing / refinement ---
from .refine import refine_with_heading_aware_astar
__all__ += ["refine_with_heading_aware_astar"]

from .router import (
    AStarParams,
    HeadingAwareAStar,
    corridor_node_mask,
    nearest_node_in_mask,
)
__all__ += ["AStarParams", "HeadingAwareAStar", "corridor_node_mask", "nearest_node_in_mask"]

# --- Optional plotting utilities (don’t fail import if plot deps missing) ---
try:
    from .plot_utils import (
        init_map,
        add_aoi_outline_from_polygon,
        add_aoi_outline,
        add_poisson_points,
        add_grid_cells,
        add_grid_cells_outline,
        add_grid_cells_filled,
        add_boundary_cells,
        fit_map_to_bounds,
    )
except Exception:
    # Plotting dependencies are optional; keep core imports usable.
    pass
else:
    __all__ += [
        "init_map",
        "add_aoi_outline_from_polygon",
        "add_aoi_outline",
        "add_poisson_points",
        "add_grid_cells",
        "add_grid_cells_outline",
        "add_grid_cells_filled",
        "add_boundary_cells",
        "fit_map_to_bounds",
    ]

