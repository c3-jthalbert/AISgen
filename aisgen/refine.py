# aisgen/refine.py
from __future__ import annotations
from typing import List, Tuple, Optional
from pyproj import Geod

import numpy as np
import networkx as nx
from shapely.geometry import LineString

from .router import (
    AStarParams,
    HeadingAwareAStar,
    corridor_node_mask,
    nearest_node_in_mask,
)


_GEOD = Geod(ellps="WGS84")

def _dist_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, d_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return d_m / 1000.0

def refine_with_heading_aware_astar(
    poisson_graph: nx.Graph,
    coarse_coords: List[Tuple[float, float]],
    speed_knots: float,
    max_turn_degps: float,
    corridor_km: float = 10.0,
    heading_bins: int = 16,
    turn_penalty_lambda: float = 0.15,
    r_min_m_override: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Refine a coarse path using heading-aware A* over the Poisson graph.
    - Snaps all coarse points to nearest Poisson nodes (preserves user intent near AOI edges).
    - Deduplicates sequential duplicates.
    - Preserves macro-route by refining each leg inside a corridor.
    - Maintains heading continuity across legs.
    """
    if poisson_graph.number_of_nodes() == 0:
        raise ValueError("Poisson graph is empty. Call env.build_poisson_graph(...) first.")

    # --- Precompute node coordinate arrays for vectorized snapping ---
    node_ids = np.array(list(poisson_graph.nodes))
    node_lons = np.array([poisson_graph.nodes[i]["lon"] for i in node_ids], dtype=float)
    node_lats = np.array([poisson_graph.nodes[i]["lat"] for i in node_ids], dtype=float)

    def snap_to_nearest(coord):
        lon, lat = coord
        # Compute distances in km to all nodes
        dists_km = np.array([_dist_km(lon, lat, nlon, nlat) for nlon, nlat in zip(node_lons, node_lats)])
        nearest_idx = np.argmin(dists_km)
        return int(node_ids[nearest_idx])

    # --- Snap all coarse points to nearest nodes ---
    snapped_ids = [snap_to_nearest(pt) for pt in coarse_coords]

    # --- Deduplicate sequential duplicates ---
    dedup_ids = [snapped_ids[0]]
    for nid in snapped_ids[1:]:
        if nid != dedup_ids[-1]:
            dedup_ids.append(nid)

    if len(dedup_ids) < 2:
        raise ValueError("Not enough unique snapped waypoints to refine path.")

    # --- Prepare router ---
    params = AStarParams(
        speed_knots=speed_knots,
        max_turn_degps=max_turn_degps,
        heading_bins=heading_bins,
        turn_penalty_lambda=turn_penalty_lambda,
        r_min_m_override=r_min_m_override,
    )
    router = HeadingAwareAStar(poisson_graph, params)

    # --- Run A* between each consecutive snapped node ---
    refined_ids = []
    prev_heading_bin = None
    for leg_start_id, leg_goal_id in zip(dedup_ids[:-1], dedup_ids[1:]):
        mask = corridor_node_mask(
            poisson_graph,
            [
                (poisson_graph.nodes[leg_start_id]["lon"], poisson_graph.nodes[leg_start_id]["lat"]),
                (poisson_graph.nodes[leg_goal_id]["lon"], poisson_graph.nodes[leg_goal_id]["lat"])
            ],
            corridor_km
        )

        # Ensure start/goal are in corridor
        if not mask[leg_start_id] or not mask[leg_goal_id]:
            mask = corridor_node_mask(
                poisson_graph,
                [
                    (poisson_graph.nodes[leg_start_id]["lon"], poisson_graph.nodes[leg_start_id]["lat"]),
                    (poisson_graph.nodes[leg_goal_id]["lon"], poisson_graph.nodes[leg_goal_id]["lat"])
                ],
                corridor_km * 2
            )

        path_ids = router.run(
            leg_start_id,
            leg_goal_id,
            allowed_nodes_mask=mask,
            initial_heading_bin=prev_heading_bin
        )

        # Append, avoiding duplicate at joins
        if refined_ids:
            refined_ids.extend(path_ids[1:])
        else:
            refined_ids.extend(path_ids)

        # Update heading continuity for next leg
        if len(path_ids) >= 2:
            u, v = path_ids[-2], path_ids[-1]
            prev_heading_bin = router.heading_bin_for_edge(u, v)

    # --- Convert final node list to coordinates ---
    coords = [
        (float(poisson_graph.nodes[i]["lon"]), float(poisson_graph.nodes[i]["lat"]))
        for i in refined_ids
    ]
    return coords
