# aisgen/utils.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import cKDTree
from pyproj import Geod
from pathlib import Path
from .environment import AOIEnvironment
from .tracks import TrackGenerator
from .vessel import load_vessel_templates



_GEOD = Geod(ellps="WGS84")


# ---------------------------
# Internal helpers
# ---------------------------

def _ensure_poisson(env) -> np.ndarray:
    """
    Ensure env.points exists; generate if missing. Returns (N,2) array (lon, lat).
    """
    if env.points is None or len(env.points) == 0:
        env.generate_poisson_points()
    return env.points


def _scaled_xy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Convert lon/lat to an approximately isotropic XY for KDTree:
      x = lon * cos(mean_lat), y = lat   (all in degrees space)

    Works well for NN search over modest extents.
    """
    if lon.size == 0:
        return np.empty((0, 2))
    lat0 = float(np.mean(lat))
    scale = np.cos(np.radians(lat0))
    return np.column_stack([lon * scale, lat])


def _densify_by_distance(coords: List[Tuple[float, float]], step_km: float) -> List[Tuple[float, float]]:
    """
    Insert points so that consecutive vertices are at most step_km apart (geodesic).
    Linear interpolation in lon/lat for placement (adequate for small steps).
    """
    if step_km is None or step_km <= 0 or len(coords) < 2:
        return coords
    out = [coords[0]]
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        _, _, d_m = _GEOD.inv(x1, y1, x2, y2)
        d_km = d_m / 1000.0
        if d_km <= step_km:
            out.append((x2, y2))
            continue
        n_add = int(np.floor(d_km / step_km))
        for i in range(1, n_add + 1):
            t = i / (n_add + 1)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        out.append((x2, y2))
    return out


# ---------------------------
# Public utilities
# ---------------------------

def snap_polyline_to_poisson(
    env,
    geojson_line: Dict[str, Any],
    densify_step_km: Optional[float] = 2.0,
    max_snap_km: Optional[float] = None,
    dedupe_consecutive: bool = True,
) -> Dict[str, Any]:
    """
    Snap every vertex of a LineString to the nearest Poisson-disk point from env.points.

    Args:
        env: AOIEnvironment with .points (lon,lat). Will call env.generate_poisson_points() if needed.
        geojson_line: GeoJSON Feature (LineString).
        densify_step_km: If set, densify the input line so vertices are ~step_km apart before snapping.
        max_snap_km: If set, any vertex whose nearest Poisson point is farther than this is dropped.
                     If too many are dropped (final < 2), raises ValueError.
        dedupe_consecutive: Drop repeated identical Poisson points in sequence.

    Returns:
        A new GeoJSON Feature (LineString) with coordinates replaced by snapped Poisson points.
        Adds properties:
          - "poisson_indices": list of indices into env.points used by the snapped path
          - "snap_meta": dict with counts and params
    """
    geom = geojson_line.get("geometry", {})
    if geom.get("type") != "LineString":
        raise ValueError("geojson_line must have geometry.type == 'LineString'")

    coords = [(float(x), float(y)) for x, y in geom["coordinates"]]
    coords = _densify_by_distance(coords, densify_step_km) if densify_step_km else coords

    pts = _ensure_poisson(env)
    if pts.size == 0:
        raise ValueError("No Poisson points available in environment.")

    # KDTree in scaled lon/lat space for speed
    pts_lon = pts[:, 0]
    pts_lat = pts[:, 1]
    tree = cKDTree(_scaled_xy(pts_lon, pts_lat))

    snapped_indices: List[int] = []
    snapped_coords: List[Tuple[float, float]] = []

    # query NN for each vertex
    q = np.array(coords, dtype=float)
    qxy = _scaled_xy(q[:, 0], q[:, 1])
    dists, idxs = tree.query(qxy, k=1)

    for (lon, lat), idx in zip(coords, idxs):
        plon, plat = float(pts_lon[idx]), float(pts_lat[idx])
        # verify geodesic distance (for max_snap_km)
        _, _, d_m = _GEOD.inv(lon, lat, plon, plat)
        d_km = d_m / 1000.0
        if max_snap_km is not None and d_km > max_snap_km:
            # skip this vertex (too far to trust)
            continue
        if dedupe_consecutive and snapped_indices and idx == snapped_indices[-1]:
            continue
        snapped_indices.append(int(idx))
        snapped_coords.append((plon, plat))

    # robust: ensure at least two vertices
    if len(snapped_coords) < 2:
        raise ValueError(
            "Snapped polyline has fewer than 2 vertices. "
            "Relax max_snap_km or increase Poisson density / densify_step_km."
        )

    out = {
        "type": "Feature",
        "properties": {
            **geojson_line.get("properties", {}),
            "poisson_indices": snapped_indices,
            "snap_meta": {
                "input_vertices": len(coords),
                "kept_vertices": len(snapped_coords),
                "densify_step_km": densify_step_km,
                "max_snap_km": max_snap_km,
            },
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [(float(x), float(y)) for x, y in snapped_coords],
        },
    }
    return out


def random_poisson_in_polygon(env, polygon: Polygon, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Pick a random Poisson point that lies inside the given polygon.
    Falls back to polygon centroid if none found.
    """
    rng = rng or np.random.default_rng()
    pts = _ensure_poisson(env)
    if pts.size == 0:
        c = polygon.centroid
        return float(c.x), float(c.y)

    # quick filter via bbox, then precise covers() for boundary-inclusive selection
    minx, miny, maxx, maxy = polygon.bounds
    mask = (pts[:, 0] >= minx) & (pts[:, 0] <= maxx) & (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
    candidates = pts[mask]
    if candidates.size:
        # precise containment
        inside = []
        for p_lon, p_lat in candidates:
            if polygon.covers(Point(float(p_lon), float(p_lat))):
                inside.append((float(p_lon), float(p_lat)))
        if inside:
            return inside[int(rng.integers(0, len(inside)))]

    # fallback
    c = polygon.centroid
    return float(c.x), float(c.y)


def setup_aisgen_environment(
    aoi_geojson: str,
    mask_geojson: Optional[str] = None,
    poisson_radius_deg: float = 0.02,
    seed: int = 42,
    target_cells: Optional[int] = 60,
    cell_width_m: Optional[float] = None,
    vessel_templates_yaml: Optional[str] = None
) -> Tuple[AOIEnvironment, TrackGenerator, Dict]:
    """
    Initialize AOIEnvironment, TrackGenerator, and vessel templates for AISgen.

    Args:
        aoi_geojson: Path to AOI GeoJSON file.
        mask_geojson: Optional mask GeoJSON to exclude (e.g., land or restricted areas).
        poisson_radius_deg: Minimum spacing for Poisson-disk sample points (in degrees).
        seed: RNG seed for reproducibility.
        target_cells: Desired number of grid cells (mutually exclusive with cell_width_m).
        cell_width_m: Desired grid cell width in meters (mutually exclusive with target_cells).
        vessel_templates_yaml: Optional path to vessel templates YAML. If None, uses the default
                               AISgen/data/vessel_class_templates.yaml.

    Returns:
        env: AOIEnvironment instance
        cells: GDF of grid partition for adjustment if necessary
        tg: TrackGenerator bound to env
        templates: dict of VesselTemplate objects
    """
    # --- Resolve default YAML path ---
    if vessel_templates_yaml is None:
        vessel_templates_yaml = Path(__file__).resolve().parent.parent / "data" / "vessel_class_templates.yaml"

    # --- Initialize environment ---
    env = AOIEnvironment.from_geojson(
        aoi_geojson,
        mask_geojson,
        poisson_radius_deg=poisson_radius_deg,
        seed=seed,
    )

    # --- Build grid partition ---
    if target_cells is not None:
        cells = env.build_grid_partition(target_cells=target_cells)
    elif cell_width_m is not None:
        cells = env.build_grid_partition(cell_width_m=cell_width_m)
    else:
        raise ValueError("Provide either target_cells or cell_width_m.")

    # --- Build graphs ---
    env.build_graph()
    env.generate_poisson_points()
    env.build_poisson_graph()

    # --- Track generator ---
    tg = TrackGenerator(env)

    # --- Vessel templates ---
    templates = load_vessel_templates(str(vessel_templates_yaml))

    return env,cells, tg, templates

