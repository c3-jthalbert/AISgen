# aisgen/refine.py
from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
from shapely.geometry import LineString

from .router import (
    AStarParams,
    HeadingAwareAStar,
    corridor_node_mask,
    nearest_node_in_mask,
)

def refine_with_heading_aware_astar(
    poisson_graph: nx.Graph,
    coarse_coords: List[Tuple[float, float]],       # [(lon,lat), ...] from Stage-1 grid path
    speed_knots: float,                             # from vessel kinematics
    max_turn_degps: float,                          # from vessel kinematics
    corridor_km: float = 10.0,                      # plan default (adjust as needed)
    heading_bins: int = 16,
    turn_penalty_lambda: float = 0.15,
    r_min_m_override: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Build a refined route inside a corridor around the coarse path,
    honoring turn-rate constraints via heading-aware A* over the Poisson graph.
    Returns a list of lon/lat coordinates (Poisson nodes along the refined path).
    """
    if poisson_graph.number_of_nodes() == 0:
        raise ValueError("Poisson graph is empty. Call env.build_poisson_graph(...) first.")

    mask = corridor_node_mask(poisson_graph, coarse_coords, corridor_km)

    # Anchor start/end to nearest corridor Poisson nodes
    start = coarse_coords[0]
    goal  = coarse_coords[-1]
    start_id = nearest_node_in_mask(poisson_graph, start, mask)
    goal_id  = nearest_node_in_mask(poisson_graph, goal,  mask)

    params = AStarParams(
        speed_knots=speed_knots,
        max_turn_degps=max_turn_degps,
        heading_bins=heading_bins,
        turn_penalty_lambda=turn_penalty_lambda,
        r_min_m_override=r_min_m_override,
    )
    router = HeadingAwareAStar(poisson_graph, params)
    path_ids = router.run(start_id, goal_id, allowed_nodes_mask=mask)

    coords = [(float(poisson_graph.nodes[i]["lon"]), float(poisson_graph.nodes[i]["lat"])) for i in path_ids]
    return coords
