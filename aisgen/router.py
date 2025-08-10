# aisgen/router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import heapq
import math

import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from .environment import _dist_km  # already defined in your code

KNOT_TO_MPS = 0.5144444444444444  # 1 kt = 0.514444.. m/s
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


# ---------------------------
# Small geo helpers
# ---------------------------

def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial great-circle bearing (degrees, 0..360)."""
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


def _gc_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return _dist_km(p1[0], p1[1], p2[0], p2[1])


def _to_utm_linebuffer(ls_wgs: LineString, buffer_km: float) -> gpd.GeoSeries:
    """Project a LineString to local UTM, return its buffer polygon, then reproject to WGS84."""
    gdf = gpd.GeoDataFrame(geometry=[ls_wgs], crs="EPSG:4326")
    utm = gdf.estimate_utm_crs()
    gdf_m = gdf.to_crs(utm)
    buf = gdf_m.buffer(buffer_km * 1000.0)  # meters
    return buf.to_crs("EPSG:4326")


# ---------------------------
# Corridor filtering on Poisson graph
# ---------------------------

def corridor_node_mask(
    Gp: nx.Graph,
    coarse_coords: List[Tuple[float, float]],
    corridor_km: float,
) -> np.ndarray:
    """
    Return boolean mask (len = #nodes in Gp in index order 0..N-1) selecting nodes
    whose point lies within `corridor_km` of the coarse path.
    NOTE: Assumes Poisson node ids are 0..N-1 contiguous (as built in environment.py).
    """
    line = LineString([(lon, lat) for lon, lat in coarse_coords])
    buf_wgs = _to_utm_linebuffer(line, corridor_km).geometry.iloc[0]

    # Build an array of node lon/lat in index order
    n = Gp.number_of_nodes()
    lons = np.empty(n, dtype=float)
    lats = np.empty(n, dtype=float)
    for i in range(n):
        d = Gp.nodes[i]
        lons[i], lats[i] = float(d["lon"]), float(d["lat"])

    # Vectorized inside test via GeoPandas (fast enough at our N)
    pts = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons, lats)], crs="EPSG:4326")
    mask = pts.within(buf_wgs) | pts.touches(buf_wgs)
    return mask.to_numpy()


# ---------------------------
# Nearest node helpers
# ---------------------------

def _scaled_xy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Roughly isotropic scaling for KDTree in lon/lat."""
    lat0 = float(np.mean(lat))
    scale = math.cos(math.radians(lat0))
    return np.column_stack([lon * scale, lat])


def nearest_node_in_mask(
    Gp: nx.Graph,
    target: Tuple[float, float],
    mask: np.ndarray,
) -> int:
    """Return node id (int) nearest to target (lon,lat) among masked nodes."""
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        raise ValueError("Corridor is empty; widen corridor_km or check coarse path.")
    lons = np.array([Gp.nodes[i]["lon"] for i in idxs], dtype=float)
    lats = np.array([Gp.nodes[i]["lat"] for i in idxs], dtype=float)
    xy = _scaled_xy(lons, lats)
    txy = _scaled_xy(np.array([target[0]]), np.array([target[1]]))[0]
    d2 = np.sum((xy - txy) ** 2, axis=1)
    j = int(np.argmin(d2))
    return int(idxs[j])


# ---------------------------
# Heading-aware A* with turn-rate constraint
# ---------------------------

@dataclass(frozen=True)
class AStarParams:
    speed_knots: float                     # vessel speed for feasibility checks
    max_turn_degps: float                  # kinematic limit
    heading_bins: int = 16                 # discretization of heading in state
    turn_penalty_lambda: float = 0.15      # weight for (Δheading)^2 cost
    # Optional: override with fixed R_min_m (else computed from speed & ω_max)
    r_min_m_override: Optional[float] = None


class HeadingAwareAStar:
    """
    Router on a Poisson graph constrained by turn-rate and preferring smoother headings.

    State: (node_id, heading_bin)
      - heading_bin is quantized last-travel heading; start has a special bin = None.

    Edge feasibility:
      - Let s be edge length (meters), v speed (m/s), ω_max (deg/s), R_min = v / (ω_max * π/180).
      - The heading change between incoming and outgoing segments must satisfy:
            |Δheading_deg| <= θ_max_deg,   θ_max_deg = (s / R_min) * 180/π
        (equivalently, |Δheading_deg| <= ω_max * (s / v))

    Cost:
      - g += s_km + λ * (Δheading_deg)^2 * 1e-3   (small quadratic penalty on heading change)
      - h  = great-circle distance to goal (km)
    """

    def __init__(self, Gp: nx.Graph, params: AStarParams):
        if Gp.number_of_nodes() == 0:
            raise ValueError("Poisson graph is empty.")
        self.Gp = Gp
        self.p = params
        self.v_mps = params.speed_knots * KNOT_TO_MPS
        self.omega_degps = params.max_turn_degps
        if params.r_min_m_override is not None:
            self.R_min_m = float(params.r_min_m_override)
        else:
            # Guard: zero turn rate => infinite R_min (straight only)
            if self.omega_degps <= 0:
                self.R_min_m = float("inf")
            else:
                self.R_min_m = self.v_mps / (self.omega_degps * DEG_TO_RAD)

        self.bin_size = 360.0 / float(max(4, params.heading_bins))

    def _bin(self, heading_deg: float) -> int:
        return int(round((heading_deg % 360.0) / self.bin_size)) % int(360.0 / self.bin_size)

    def _bin_center(self, b: int) -> float:
        return (b + 0.5) * self.bin_size

    def _edge_len_km(self, u: int, v: int) -> float:
        nu, nv = self.Gp.nodes[u], self.Gp.nodes[v]
        return _gc_km((nu["lon"], nu["lat"]), (nv["lon"], nv["lat"]))

    def _bearing_uv(self, u: int, v: int) -> float:
        nu, nv = self.Gp.nodes[u], self.Gp.nodes[v]
        return _bearing_deg(nu["lon"], nu["lat"], nv["lon"], nv["lat"])

    def _feasible_turn(self, d_heading_deg: float, s_km: float) -> bool:
        if math.isinf(self.R_min_m):
            # No turning allowed (omega==0): only accept zero heading change
            return abs(d_heading_deg) < 1e-6
        s_m = s_km * 1000.0
        theta_max_deg = (s_m / self.R_min_m) * RAD_TO_DEG
        # Also enforce ω_max * (s / v) directly (redundant but numerically stable)
        theta_max_deg_time = self.omega_degps * (s_m / max(self.v_mps, 1e-6))
        theta_cap = min(theta_max_deg, theta_max_deg_time)
        return abs(d_heading_deg) <= (theta_cap + 1e-9)

    def run(
        self,
        start_node: int,
        goal_node: int,
        allowed_nodes_mask: Optional[np.ndarray] = None,
        max_expansions: int = 500000,
    ) -> List[int]:
        """Return a list of node ids from start to goal (inclusive). Raises if not found."""
        if allowed_nodes_mask is not None:
            if not allowed_nodes_mask[start_node] or not allowed_nodes_mask[goal_node]:
                raise ValueError("Start/goal not in corridor; widen corridor_km or re-anchor.")
        Nbins = int(360.0 / self.bin_size)

        # Priority queue of (f, g, node, heading_bin_or_None)
        openq: List[Tuple[float, float, int, Optional[int]]] = []
        # g-costs and backpointers keyed by (node, bin)
        gbest: Dict[Tuple[int, Optional[int]], float] = {}
        parent: Dict[Tuple[int, Optional[int]], Tuple[int, Optional[int]]] = {}

        def h(node: int) -> float:
            # straight-line great-circle distance to goal in km
            n1, n2 = self.Gp.nodes[node], self.Gp.nodes[goal_node]
            return _gc_km((n1["lon"], n1["lat"]), (n2["lon"], n2["lat"]))

        start_state = (start_node, None)  # None means "no prior heading"
        gbest[start_state] = 0.0
        heapq.heappush(openq, (h(start_node), 0.0, start_node, None))

        expansions = 0
        visited = set()

        while openq:
            f, g, u, hb = heapq.heappop(openq)
            state = (u, hb)
            if state in visited:
                continue
            visited.add(state)

            if u == goal_node:
                # reconstruct path by ignoring heading bins (take the first goal we pop)
                path_nodes: List[int] = [u]
                cur = state
                while cur in parent:
                    cur = parent[cur]
                    path_nodes.append(cur[0])
                path_nodes.reverse()
                return path_nodes

            if expansions > max_expansions:
                break
            expansions += 1

            # Neighbor expansion
            for v in self.Gp.neighbors(u):
                if allowed_nodes_mask is not None and not allowed_nodes_mask[v]:
                    continue
                s_km = self._edge_len_km(u, v)
                if s_km <= 1e-6:
                    continue

                new_heading = self._bearing_uv(u, v)
                # If we have a prior heading bin, enforce feasibility and compute penalty
                if hb is None:
                    d_heading = 0.0
                    penalty = 0.0
                    nb = self._bin(new_heading)
                else:
                    prior_heading = self._bin_center(hb)
                    d_heading = _angdiff_deg(prior_heading, new_heading)
                    if not self._feasible_turn(d_heading, s_km):
                        continue
                    penalty = self.p.turn_penalty_lambda * (d_heading * d_heading) * 1e-3
                    nb = self._bin(new_heading)

                g2 = g + s_km + penalty
                st2 = (v, nb)
                if g2 < gbest.get(st2, float("inf")):
                    gbest[st2] = g2
                    parent[st2] = state
                    f2 = g2 + h(v)
                    heapq.heappush(openq, (f2, g2, v, nb))

        raise RuntimeError("A* failed to find a path under the given constraints.")
