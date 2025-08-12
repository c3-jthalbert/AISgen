# aisgen/tracks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import networkx as nx
from pyproj import Geod

from .environment import AOIEnvironment
from .vessel import VesselTemplate, VesselInstance
from .refine import refine_with_heading_aware_astar  # Stage-2 router

# ---------------------
# Shared utilities
# ---------------------

_GEOD = Geod(ellps="WGS84")


def _dist_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, d_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return d_m / 1000.0


def _timestamp_waypoints(
    coords: List[Tuple[float, float]],
    speed_knots: float,
    start_time: Optional[datetime],
    track_id: str,
) -> pd.DataFrame:
    if start_time is None:
        start_time = datetime.utcnow()

    speed_kmh = max(float(speed_knots) * 1.852, 0.1)
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


def _densify_by_time(coords: List[Tuple[float, float]], speed_knots: float, step_minutes: int = 5) -> List[Tuple[float, float]]:
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
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dλ = math.radians(lon2 - lon1)
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360.0) % 360.0


def _angdiff_deg(a: float, b: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return d if d != -180.0 else 180.0


def _sample_polyline_constant_speed(
    coords: List[Tuple[float, float]],
    speed_knots: float,
    sample_dt_s: int,
    start_time: Optional[datetime],
    track_id: str,
) -> pd.DataFrame:
    if len(coords) < 2:
        return _timestamp_waypoints(coords, speed_knots, start_time, track_id)

    v_mps = max(float(speed_knots) * 0.514444, 0.1)
    if start_time is None:
        start_time = datetime.utcnow()

    seg_m = []
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        _, _, d_m = _GEOD.inv(x1, y1, x2, y2)
        seg_m.append(max(d_m, 0.0))
    cum_m = np.concatenate([[0.0], np.cumsum(seg_m)])

    total_t = float(cum_m[-1] / v_mps)
    n_steps = max(1, int(math.floor(total_t / float(sample_dt_s))) + 1)

    times, lons, lats = [], [], []
    for k in range(n_steps):
        tsec = min(k * sample_dt_s, total_t)
        s = tsec * v_mps
        j = int(np.searchsorted(cum_m, s, side="right") - 1)
        j = max(0, min(j, len(coords) - 2))
        s0, s1 = cum_m[j], cum_m[j + 1]
        alpha = 0.0 if s1 - s0 <= 1e-6 else (s - s0) / (s1 - s0)
        (x1, y1), (x2, y2) = coords[j], coords[j + 1]
        lon = x1 + alpha * (x2 - x1)
        lat = y1 + alpha * (y2 - y1)
        times.append(start_time + timedelta(seconds=float(tsec)))
        lons.append(float(lon))
        lats.append(float(lat))

    headings = [_bearing_deg(lons[i], lats[i], lons[i + 1], lats[i + 1])
                if i + 1 < len(lons) else _bearing_deg(lons[i - 1], lats[i - 1], lons[i], lats[i])
                for i in range(len(lons))]

    dt = float(sample_dt_s)
    turn = []
    for i in range(len(headings)):
        if i == 0 or i == len(headings) - 1 or dt <= 0:
            turn.append(0.0)
        else:
            d1 = _angdiff_deg(headings[i - 1], headings[i])
            d2 = _angdiff_deg(headings[i], headings[i + 1])
            turn.append((d1 + d2) / (2.0 * dt))

    return pd.DataFrame({
        "TrackID": track_id,
        "Timestamp": pd.to_datetime(times),
        "Longitude": lons,
        "Latitude": lats,
        "Speed_knots": float(speed_knots),
        "Heading_deg": headings,
        "TurnRate_degps": turn,
        "SegmentIndex": 0,
        "Stage": "sampled",
    })


def _emit_geojson_minimal(coords: List[Tuple[float, float]], track_id: str) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "properties": {"TrackID": track_id, "generator_version": "aisgen-2.1"},
        "geometry": {"type": "LineString", "coordinates": [(float(x), float(y)) for x, y in coords]},
    }


def _refine_then_sample(
    env: AOIEnvironment,
    vessel: VesselInstance,
    coarse_coords: List[Tuple[float, float]],
    speed_knots: Optional[float],
    corridor_km: float,
    heading_bins: int,
    straightness_penalty: float,
    sample_dt_s: int,
    start_time: Optional[datetime],
    track_id: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    v_kn = float(speed_knots) if speed_knots is not None else vessel.kinematics.get("cruise_speed_kn", 10.0)
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
    df = _sample_polyline_constant_speed(refined_coords, v_kn, int(sample_dt_s), start_time, track_id)
    for k, v in vessel.kinematics.items():
        df[k] = v
    for k, v in vessel.metadata.items():
        #print(vessel.metadata.items())
        df[k] = v
    df["emitter_profile"] = [vessel.emitter_profile] * len(df)
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
    env: AOIEnvironment
    rng: np.random.Generator = np.random.default_rng(42)

    def random_track(self, vessel_template: VesselTemplate, **kwargs) -> Tuple[Dict[str, Any], pd.DataFrame]:
        vessel = vessel_template.sample(int(self.rng.integers(0, 2**32 - 1)))
        if getattr(vessel, "emitter_profile", None) is None:
            raise ValueError(f"Emitter profile missing right after sampling from {vessel_template.name}")
        return self._random_track_instance(vessel, **kwargs)

    def _random_track_instance(
        self,
        vessel: VesselInstance,
        n_waypoint_cells: int = 4,
        prefer_ports: bool = True,
        start_time: Optional[datetime] = None,
        speed_knots: Optional[float] = None,
        min_cells_apart: int = 5,
        track_id: Optional[str] = None,
        refine: bool = True,
        corridor_km: float = 10.0,
        heading_bins: int = 16,
        straightness_penalty: float = 0.05,
        sample_dt_s: int = 20,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        cells = self.env.partition_gdf
        if cells is None or self.env.graph is None:
            raise ValueError("Environment must have grid and graph built.")

        if prefer_ports and "is_port" in cells.columns and cells["is_port"].any():
            start_candidates = cells.loc[cells["is_port"], "cell_id"].tolist()
        elif "is_boundary" in cells.columns and cells["is_boundary"].any():
            start_candidates = cells.loc[cells["is_boundary"], "cell_id"].tolist()
        else:
            start_candidates = cells["cell_id"].tolist()
        start_cell = int(self.rng.choice(start_candidates))

        end_candidates = [cid for cid in cells["cell_id"].tolist() if abs(cid - start_cell) >= min_cells_apart]
        if not end_candidates:
            raise ValueError("No valid end cell found.")
        end_cell = int(self.rng.choice(end_candidates))

        #path_cells = nx.shortest_path(self.env.graph, start_cell, end_cell, weight="weight")
        #coarse_coords = [(cells.loc[cells["cell_id"] == cid, "centroid_x"].values[0],
        #                  cells.loc[cells["cell_id"] == cid, "centroid_y"].values[0]) for cid in path_cells]
        path_cells = nx.shortest_path(self.env.graph, start_cell, end_cell, weight="weight")
        coarse_coords = []
        for cid in path_cells:
            lon, lat = self.env.sample_poisson_point_in_cell(int(cid), rng=self.rng)
            coarse_coords.append((float(lon), float(lat)))

        if not refine:
            geo_min = _emit_geojson_minimal(coarse_coords, track_id or "rand")
            df = _sample_polyline_constant_speed(coarse_coords, speed_knots or vessel.kinematics.get("cruise_speed_kn", 10.0), sample_dt_s, start_time, track_id or "rand")
            for k, v in vessel.kinematics.items():
                df[k] = v
            for k, v in vessel.metadata.items():
                df[k] = v
            return geo_min, df

        return _refine_then_sample(self.env, vessel, coarse_coords, speed_knots, corridor_km, heading_bins, straightness_penalty, sample_dt_s, start_time, track_id or "rand")

    def from_geojson(
        self,
        vessel_template: VesselTemplate,
        geojson_line: Dict[str, Any],
        refine: bool = True,
        **kwargs
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        vessel = vessel_template.sample(int(self.rng.integers(0, 2**32 - 1)))
        coords = geojson_line["geometry"]["coordinates"]
        coarse_coords = [(float(x), float(y)) for x, y in coords]
        if not refine:
            geo_min = _emit_geojson_minimal(coarse_coords, kwargs.get("track_id", "geo"))
            df = _sample_polyline_constant_speed(coarse_coords, kwargs.get("speed_knots", vessel.kinematics.get("cruise_speed_kn", 10.0)), kwargs.get("sample_dt_s", 20), kwargs.get("start_time", None), kwargs.get("track_id", "geo"))
            for k, v in vessel.kinematics.items():
                df[k] = v
            for k, v in vessel.metadata.items():
                df[k] = v
            return geo_min, df
        return _refine_then_sample(self.env, vessel, coarse_coords, kwargs.get("speed_knots", None), kwargs.get("corridor_km", 10.0), kwargs.get("heading_bins", 16), kwargs.get("straightness_penalty", 0.05), kwargs.get("sample_dt_s", 20), kwargs.get("start_time", None), kwargs.get("track_id", "geo"))
