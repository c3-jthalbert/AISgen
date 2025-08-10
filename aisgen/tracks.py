# aisgen/tracks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import networkx as nx
from pyproj import Geod

from .environment import AOIEnvironment
from .vessel import VesselTemplate

import math
from .refine import refine_with_heading_aware_astar  # Stage‑2 router

# ---------------------
# Shared utilities
# ---------------------

_GEOD = Geod(ellps="WGS84")


def _dist_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Geodesic distance in kilometers."""
    _, _, d_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return d_m / 1000.0


def _timestamp_waypoints(
    coords: List[Tuple[float, float]],
    speed_knots: float,
    start_time: Optional[datetime],
    track_id: str,
) -> pd.DataFrame:
    """
    Assign timestamps to the given waypoints at constant speed.
    One row per waypoint (no densification).
    """
    if start_time is None:
        start_time = datetime.utcnow()

    # knots -> km/h (1 knot = 1.852 km/h)
    speed_kmh = max(float(speed_knots) * 1.852, 0.1)  # avoid zero
    seg_km = [0.0]
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        seg_km.append(_dist_km(x1, y1, x2, y2))
    hours = np.cumsum([d / speed_kmh for d in seg_km])

    timestamps = [start_time + timedelta(hours=float(h)) for h in hours]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return pd.DataFrame(
        {
            "TrackID": track_id,
            "Timestamp": pd.to_datetime(timestamps),
            "Longitude": lons,
            "Latitude": lats,
            "Speed_knots": float(speed_knots),
        }
    )


def _densify_by_time(
    coords: List[Tuple[float, float]],
    speed_knots: float,
    step_minutes: int = 5,
) -> List[Tuple[float, float]]:
    """Insert intermediate points so consecutive vertices are ~speed*step apart."""
    if len(coords) < 2 or step_minutes <= 0:
        return coords
    speed_kmh = max(float(speed_knots) * 1.852, 0.1)
    target_km = speed_kmh * (step_minutes / 60.0)
    out = [coords[0]]
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        seg = _dist_km(x1, y1, x2, y2)
        if seg <= 1e-6:
            continue
        n_add = int(np.floor(seg / target_km))
        for i in range(1, n_add + 1):
            t = i / (n_add + 1)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        out.append((x2, y2))
    return out

def _bearing_deg(lon1, lat1, lon2, lat2) -> float:
    """Initial great‑circle bearing (degrees, 0..360)."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360.0) % 360.0

def _angdiff_deg(a: float, b: float) -> float:
    """Smallest signed difference a->b in degrees in (-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return d if d != -180.0 else 180.0

def _sample_polyline_constant_speed(
    coords: List[Tuple[float, float]],
    speed_knots: float,
    sample_dt_s: int,
    start_time: Optional[datetime],
    track_id: str,
) -> pd.DataFrame:
    """
    Walk a polyline at constant speed, emitting samples every sample_dt_s.
    Adds Heading_deg and TurnRate_degps (finite diff over time).
    """
    if len(coords) < 2:
        return _timestamp_waypoints(coords, speed_knots, start_time, track_id)

    v_mps = float(speed_knots) * 0.514444
    if v_mps <= 0:
        v_mps = 0.1
    if start_time is None:
        start_time = datetime.utcnow()

    # Build cumulative arc-length (meters) along edges
    seg_m = []
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        _, _, d_m = _GEOD.inv(x1, y1, x2, y2)
        seg_m.append(max(d_m, 0.0))
    cum_m = np.concatenate([[0.0], np.cumsum(seg_m)])

    total_t = float(cum_m[-1] / v_mps)
    n_steps = max(1, int(math.floor(total_t / float(sample_dt_s))) + 1)

    # For each t_k, interpolate lon/lat on the polyline by arc-length fraction
    times = []
    lons = []
    lats = []
    for k in range(n_steps):
        tsec = min(k * sample_dt_s, total_t)
        s = tsec * v_mps  # meters
        # locate segment
        j = int(np.searchsorted(cum_m, s, side="right") - 1)
        j = max(0, min(j, len(coords) - 2))
        s0, s1 = cum_m[j], cum_m[j + 1]
        if s1 - s0 <= 1e-6:
            alpha = 0.0
        else:
            alpha = (s - s0) / (s1 - s0)
        (x1, y1), (x2, y2) = coords[j], coords[j + 1]
        lon = x1 + alpha * (x2 - x1)
        lat = y1 + alpha * (y2 - y1)

        times.append(start_time + timedelta(seconds=float(tsec)))
        lons.append(float(lon))
        lats.append(float(lat))

    # Headings at samples
    headings = [ _bearing_deg(lons[i], lats[i], lons[i+1], lats[i+1])
                 if i+1 < len(lons) else _bearing_deg(lons[i-1], lats[i-1], lons[i], lats[i])
                 for i in range(len(lons)) ]

    # Turn rate via centered difference on heading
    dt = float(sample_dt_s)
    turn = []
    for i in range(len(headings)):
        if i == 0 or i == len(headings) - 1 or dt <= 0:
            turn.append(0.0)
        else:
            d1 = _angdiff_deg(headings[i-1], headings[i])
            d2 = _angdiff_deg(headings[i], headings[i+1])
            # average slope (deg/s)
            turn.append( (d1 + d2) / (2.0 * dt) )

    df = pd.DataFrame({
        "TrackID": track_id,
        "Timestamp": pd.to_datetime(times),
        "Longitude": lons,
        "Latitude": lats,
        "Speed_knots": float(speed_knots),
        "Heading_deg": headings,
        "TurnRate_degps": turn,
        "SegmentIndex": 0,     # placeholder until we add straight/arc primitives
        "Stage": "sampled",
    })
    return df

def _emit_geojson_minimal(coords: List[tuple[float,float]], track_id: str) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"TrackID": track_id, "generator_version": "aisgen-2.1"},
        "geometry": {"type": "LineString", "coordinates": [(float(x), float(y)) for x, y in coords]},
    }

def _refine_then_sample(
    env: AOIEnvironment,
    vessel: VesselTemplate,
    coarse_coords: List[tuple[float,float]],
    speed_knots: Optional[float],
    corridor_km: float,
    heading_bins: int,
    straightness_penalty: float,
    sample_dt_s: int,
    start_time: Optional[datetime],
    track_id: str,
) -> tuple[Dict[str,Any], pd.DataFrame]:
    v_kn = float(speed_knots) if speed_knots is not None else vessel.sample_speed_knots()
    max_turn = float(vessel.kinematics.get("max_turn_deg_per_s", 3.0))
    refined_coords = refine_with_heading_aware_astar(
        poisson_graph=env.poisson_graph,
        coarse_coords=coarse_coords,
        speed_knots=v_kn,
        max_turn_degps=max_turn,
        corridor_km=float(corridor_km),
        heading_bins=int(heading_bins),
        turn_penalty_lambda=float(straightness_penalty),
    )
    geo_min = _emit_geojson_minimal(refined_coords, track_id)
    df = _sample_polyline_constant_speed(
        coords=refined_coords,
        speed_knots=v_kn,
        sample_dt_s=int(sample_dt_s),
        start_time=start_time,
        track_id=track_id,
    )
    # repeat vessel & routing fields
    for k, v in vessel.kinematics.items():
        df[k] = v
    df["poisson_radius_deg"] = float(env.poisson_radius_deg)
    df["corridor_km"] = float(corridor_km)
    df["heading_bins"] = int(heading_bins)
    df["straightness_penalty"] = float(straightness_penalty)
    df["sample_dt_s"] = int(sample_dt_s)
    df["generator_version"] = "aisgen-2.1"
    return geo_min, df


# ---------------------
# Track generators
# ---------------------

@dataclass
class TrackGenerator:
    """
    Generates vessel tracks using AOIEnvironment and VesselTemplate.
    - Picks start/end cells (prefers ports, then boundary)
    - Walks a path through the grid graph
    - Samples one point per chosen cell (from env Poisson points or centroid fallback)
    - Builds a GeoJSON LineString and a timestamped waypoint DataFrame
    """
    env: AOIEnvironment
    rng: np.random.Generator = np.random.default_rng(42)

    def random_track(
        self,
        vessel_template: VesselTemplate,
        n_waypoint_cells: int = 4,
        prefer_ports: bool = True,
        start_time: Optional[datetime] = None,
        speed_knots: Optional[float] = None,
        min_cells_apart: int = 5,
        track_id: Optional[str] = None,
        # new:
        refine: bool = False,
        corridor_km: float = 10.0,
        heading_bins: int = 16,
        straightness_penalty: float = 0.05,
        sample_dt_s: int = 20,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:

        if self.env.partition_gdf is None or self.env.graph is None:
            raise ValueError(
                "Environment must have a grid partition and graph. "
                "Call env.build_grid_partition(...); env.build_graph()."
            )
        
        G = self.env.graph
        cells = self.env.partition_gdf
        
        # --- choose start cell ---
        if prefer_ports and "is_port" in cells.columns and cells["is_port"].any():
            start_candidates = cells.loc[cells["is_port"], "cell_id"].tolist()
        elif "is_boundary" in cells.columns and cells["is_boundary"].any():
            start_candidates = cells.loc[cells["is_boundary"], "cell_id"].tolist()
        else:
            start_candidates = cells["cell_id"].tolist()
        start_cell = int(self.rng.choice(start_candidates))
        
        # --- choose end cell far enough away on the graph ---
        lengths = nx.single_source_shortest_path_length(G, start_cell)
        far = [cid for cid, L in lengths.items() if L >= max(min_cells_apart, 2)]
        if prefer_ports and "is_port" in cells.columns and cells["is_port"].any():
            far_ports = list(set(far) & set(cells.loc[cells["is_port"], "cell_id"]))
            far = far_ports or far
        elif "is_boundary" in cells.columns and cells["is_boundary"].any():
            far_bnd = list(set(far) & set(cells.loc[cells["is_boundary"], "cell_id"]))
            far = far_bnd or far
        end_cell = int(self.rng.choice(far)) if far else int(self.rng.choice(cells["cell_id"]))
        
        # --- shortest path (fallback: random walk) ---
        try:
            path = nx.shortest_path(G, start_cell, end_cell)
        except nx.NetworkXNoPath:
            path = [start_cell]
            cur = start_cell
            for _ in range(n_waypoint_cells + 1):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                nxt = int(self.rng.choice(nbrs))
                if nxt != path[-1]:
                    path.append(nxt)
                    cur = nxt
        
        # downsample to ~n_waypoint_cells interiors
        if len(path) > (n_waypoint_cells + 2):
            interior = path[1:-1]
            if n_waypoint_cells > 0:
                idx = np.linspace(0, len(interior) - 1, num=n_waypoint_cells, dtype=int).tolist()
                chosen = [interior[i] for i in idx]
                cell_path = [path[0]] + chosen + [path[-1]]
            else:
                cell_path = [path[0], path[-1]]
        else:
            cell_path = path

        coords: List[Tuple[float, float]] = [
            self.env.sample_poisson_point_in_cell(int(cid), rng=self.rng) for cid in cell_path
        ]
        
        tid = track_id or f"rand_{start_cell}_{end_cell}"
        if not refine:
            spd_kn = float(speed_knots) if speed_knots is not None else vessel_template.sample_speed_knots(self.rng)
            geojson_line = {
                "type": "Feature",
                "properties": {"TrackID": tid, "vessel_type": vessel_template.vessel_type},
                "geometry": {"type": "LineString", "coordinates": [(float(lon), float(lat)) for lon, lat in coords]},
            }
            track_df = _timestamp_waypoints(coords, spd_kn, start_time, tid)
            return geojson_line, track_df
    
        # refine path using Poisson corridor + heading-aware A*
        if self.env.poisson_graph is None or self.env.poisson_graph.number_of_nodes() == 0:
            raise ValueError("Poisson graph missing. Call env.generate_poisson_points(); env.build_poisson_graph().")
    
        geo_min, df = _refine_then_sample(
            env=self.env,
            vessel=vessel_template,
            coarse_coords=coords,
            speed_knots=speed_knots,
            corridor_km=corridor_km,
            heading_bins=heading_bins,
            straightness_penalty=straightness_penalty,
            sample_dt_s=sample_dt_s,
            start_time=start_time,
            track_id=tid,
        )
        return geo_min, df

    def from_geojson(
        self,
        vessel_template: VesselTemplate,
        geojson_line: Dict[str, Any],
        start_time: Optional[datetime] = None,
        speed_knots: Optional[float] = None,
        track_id: Optional[str] = None,
        # new:
        refine: bool = False,
        corridor_km: float = 10.0,
        heading_bins: int = 16,
        straightness_penalty: float = 0.05,
        sample_dt_s: int = 20,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        geom = geojson_line.get("geometry", {})
        if geom.get("type") != "LineString":
            raise ValueError("geojson_line must have geometry.type == 'LineString'")
        coords = [(float(x), float(y)) for x, y in geom["coordinates"]]
        props = geojson_line.setdefault("properties", {})
        tid = track_id or props.get("TrackID") or props.get("track_id") or "geojson_track"
        props["TrackID"] = tid
        props.setdefault("vessel_type", vessel_template.vessel_type)
    
        if not refine:
            spd_kn = float(speed_knots) if speed_knots is not None else vessel_template.sample_speed_knots(self.rng)
            df = _timestamp_waypoints(coords, spd_kn, start_time, tid)
            return geojson_line, df
    
        if self.env.poisson_graph is None or self.env.poisson_graph.number_of_nodes() == 0:
            raise ValueError("Poisson graph missing. Call env.generate_poisson_points(); env.build_poisson_graph().")
    
        geo_min, df = _refine_then_sample(
            env=self.env,
            vessel=vessel_template,
            coarse_coords=coords,
            speed_knots=speed_knots,
            corridor_km=corridor_km,
            heading_bins=heading_bins,
            straightness_penalty=straightness_penalty,
            sample_dt_s=sample_dt_s,
            start_time=start_time,
            track_id=tid,
        )
        return geo_min, df

    def coarse_then_refined(
        self,
        vessel: VesselTemplate,
        prefer_ports: bool = True,
        corridor_km: float = 10.0,
        heading_bins: int = 16,
        straightness_penalty: float = 0.05,
        speed_knots: Optional[float] = None,
        sample_dt_s: int = 20,
        rng_seed: Optional[int] = None,
        track_id: Optional[str] = None,
        n_waypoint_cells: int = 8,
        min_cells_apart: int = 5,
        start_time: Optional[datetime] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Stage‑1 (grid) → Stage‑2 (Poisson refinement w/ turn limits) → per‑point sampling.
        Returns (minimal GeoJSON, rich per‑point DataFrame).
        """
        if self.env.partition_gdf is None or self.env.graph is None:
            raise ValueError("Environment missing grid/graph. Call env.build_grid_partition(); env.build_graph().")

        if self.env.poisson_graph is None or self.env.poisson_graph.number_of_nodes() == 0:
            raise ValueError("Poisson graph missing. Call env.generate_poisson_points(); env.build_poisson_graph().")

        if rng_seed is not None:
            self.rng = np.random.default_rng(int(rng_seed))

        # --- Stage 1: coarse grid path (reuse existing logic) ---
        G = self.env.graph
        cells = self.env.partition_gdf

        # choose start/end (ports -> boundary -> anywhere)
        if prefer_ports and "is_port" in cells.columns and cells["is_port"].any():
            start_candidates = cells.loc[cells["is_port"], "cell_id"].tolist()
        elif "is_boundary" in cells.columns and cells["is_boundary"].any():
            start_candidates = cells.loc[cells["is_boundary"], "cell_id"].tolist()
        else:
            start_candidates = cells["cell_id"].tolist()
        start_cell = int(self.rng.choice(start_candidates))

        lengths = nx.single_source_shortest_path_length(G, start_cell)
        far = [cid for cid, L in lengths.items() if L >= max(min_cells_apart, 2)]
        if prefer_ports and "is_port" in cells.columns and cells["is_port"].any():
            far_ports = list(set(far) & set(cells.loc[cells["is_port"], "cell_id"]))
            far = far_ports or far
        elif "is_boundary" in cells.columns and cells["is_boundary"].any():
            far_bnd = list(set(far) & set(cells.loc[cells["is_boundary"], "cell_id"]))
            far = far_bnd or far
        end_cell = int(self.rng.choice(far)) if far else int(self.rng.choice(cells["cell_id"]))

        try:
            path = nx.shortest_path(G, start_cell, end_cell)
        except nx.NetworkXNoPath:
            path = [start_cell, end_cell]

        # downsample interiors to ~n_waypoint_cells
        if len(path) > (n_waypoint_cells + 2):
            interior = path[1:-1]
            n_in = max(n_waypoint_cells, 0)
            idx = np.linspace(0, len(interior) - 1, num=n_in, dtype=int).tolist()
            cell_path = [path[0]] + [interior[i] for i in idx] + [path[-1]]
        else:
            cell_path = path

        # one point per chosen cell
        coarse_coords: List[Tuple[float, float]] = [
            self.env.sample_poisson_point_in_cell(int(cid), rng=self.rng) for cid in cell_path
        ]

        # --- Stage 2: heading‑aware A* inside corridor (turn‑limited) ---
        # pick planning speed & ω_max from vessel kinematics
        v_kn = float(speed_knots) if speed_knots is not None else vessel.sample_speed_knots(self.rng)
        max_turn = float(vessel.kinematics.get("max_turn_deg_per_s", 3.0))
        refined_coords = refine_with_heading_aware_astar(
            poisson_graph=self.env.poisson_graph,
            coarse_coords=coarse_coords,
            speed_knots=v_kn,
            max_turn_degps=max_turn,
            corridor_km=float(corridor_km),
            heading_bins=int(heading_bins),
            turn_penalty_lambda=float(straightness_penalty),
        )

        # --- minimal GeoJSON (per plan) ---
        tid = track_id or f"refined_{start_cell}_{end_cell}"
        geo_minimal = {
            "type": "Feature",
            "properties": {"TrackID": tid, "generator_version": "aisgen-2.1"},
            "geometry": {"type": "LineString", "coordinates": [(float(x), float(y)) for x, y in refined_coords]},
        }

        # --- per‑point sampling with headings/turn rate ---
        df = _sample_polyline_constant_speed(
            coords=refined_coords,
            speed_knots=v_kn,
            sample_dt_s=int(sample_dt_s),
            start_time=start_time,
            track_id=tid,
        )

        # repeat vessel & routing fields across rows (authoritative table)
        # (metadata could be large; attach kinematics + a few routing knobs for now)
        for k, v in vessel.kinematics.items():
            df[k] = v
        df["poisson_radius_deg"] = float(self.env.poisson_radius_deg)
        df["corridor_km"] = float(corridor_km)
        df["heading_bins"] = int(heading_bins)
        df["straightness_penalty"] = float(straightness_penalty)
        df["sample_dt_s"] = int(sample_dt_s)
        df["rng_seed"] = int(rng_seed) if rng_seed is not None else None
        df["generator_version"] = "aisgen-2.1"

        return geo_minimal, df

    def from_coarse_geojson_with_refinement(
        self,
        vessel: VesselTemplate,
        coarse_geojson: Dict[str, Any],
        corridor_km: float = 10.0,
        heading_bins: int = 16,
        straightness_penalty: float = 0.05,
        speed_knots: Optional[float] = None,
        sample_dt_s: int = 20,
        start_time: Optional[datetime] = None,
        track_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Stage‑2 only: given a user coarse LineString, refine with heading‑aware A* and sample.
        """
        if self.env.poisson_graph is None or self.env.poisson_graph.number_of_nodes() == 0:
            raise ValueError("Poisson graph missing. Call env.generate_poisson_points(); env.build_poisson_graph().")

        geom = coarse_geojson.get("geometry", {})
        if geom.get("type") != "LineString":
            raise ValueError("coarse_geojson must be a LineString")
        coarse_coords = [(float(x), float(y)) for x, y in geom["coordinates"]]

        v_kn = float(speed_knots) if speed_knots is not None else vessel.sample_speed_knots(self.rng)
        max_turn = float(vessel.kinematics.get("max_turn_deg_per_s", 3.0))

        refined_coords = refine_with_heading_aware_astar(
            poisson_graph=self.env.poisson_graph,
            coarse_coords=coarse_coords,
            speed_knots=v_kn,
            max_turn_degps=max_turn,
            corridor_km=float(corridor_km),
            heading_bins=int(heading_bins),
            turn_penalty_lambda=float(straightness_penalty),
        )

        tid = track_id or coarse_geojson.get("properties", {}).get("TrackID", "refined_from_user")
        geo_minimal = {
            "type": "Feature",
            "properties": {"TrackID": tid, "generator_version": "aisgen-2.1"},
            "geometry": {"type": "LineString", "coordinates": [(float(x), float(y)) for x, y in refined_coords]},
        }

        df = _sample_polyline_constant_speed(
            coords=refined_coords,
            speed_knots=v_kn,
            sample_dt_s=int(sample_dt_s),
            start_time=start_time,
            track_id=tid,
        )

        for k, v in vessel.kinematics.items():
            df[k] = v
        df["poisson_radius_deg"] = float(self.env.poisson_radius_deg)
        df["corridor_km"] = float(corridor_km)
        df["heading_bins"] = int(heading_bins)
        df["straightness_penalty"] = float(straightness_penalty)
        df["sample_dt_s"] = int(sample_dt_s)
        df["generator_version"] = "aisgen-2.1"

        return geo_minimal, df

@dataclass
class TrackBuilder:
    """
    Build a timestamped track from a provided polyline, with optional time-based densification.
    """
    rng: np.random.Generator = np.random.default_rng(42)

    def from_polyline(
        self,
        vessel_template: VesselTemplate,
        geojson_line: Dict[str, Any],
        start_time: Optional[datetime] = None,
        speed_knots: Optional[float] = None,
        track_id: Optional[str] = None,
        densify_minutes: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        geom = geojson_line.get("geometry", {})
        if geom.get("type") != "LineString":
            raise ValueError("geojson_line must be a LineString")
        coords = [(float(x), float(y)) for x, y in geom["coordinates"]]

        tid = track_id or geojson_line.get("properties", {}).get("track_id", "poly_track")
        spd_kn = float(speed_knots) if speed_knots is not None else vessel_template.sample_speed_knots(self.rng)

        if densify_minutes and densify_minutes > 0:
            coords = _densify_by_time(coords, spd_kn, step_minutes=densify_minutes)

        df = _timestamp_waypoints(coords, spd_kn, start_time, tid)
        props = geojson_line.setdefault("properties", {})
        props.setdefault("track_id", tid)
        props.setdefault("vessel_type", vessel_template.vessel_type)
        return geojson_line, df
