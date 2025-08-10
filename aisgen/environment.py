# aisgen/environment.py
"""
AOIEnvironment (projected-CRS safe):
- AOI polygon (water-only), stored in EPSG:4326 (lon/lat)
- Poisson-disk points (lon/lat)
- Fixed rectangular grid partition (independent of points)
- Cell adjacency graph with centroids computed in a projected CRS (no warnings)

Notes:
- We project to an estimated local UTM for any operation that depends on linear/area
  accuracy (grid construction, area filters, centroids for graph nodes), then convert
  results back to EPSG:4326 for storage/return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Union

import numpy as np
import geopandas as gpd
import networkx as nx

from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from scipy.stats.qmc import PoissonDisk
from pyproj import Geod
from scipy.spatial import cKDTree


# ---------------------------
# Internal helpers
# ---------------------------

def _to_utm(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    """
    Reproject a GeoDataFrame from EPSG:4326 to an estimated local UTM.
    Returns (reprojected_gdf, utm_epsg_string).
    """
    utm = gdf.estimate_utm_crs()
    return gdf.to_crs(utm), str(utm)


def _grid_bounds(minx: float, miny: float, maxx: float, maxy: float,
                 dx: float, dy: float) -> Iterable[Polygon]:
    """
    Yield rectangular cells (as shapely boxes) covering the bbox with spacing dx, dy.
    """
    xs = np.arange(minx, maxx + dx, dx)
    ys = np.arange(miny, maxy + dy, dy)
    for x0 in xs[:-1]:
        for y0 in ys[:-1]:
            yield box(x0, y0, x0 + dx, y0 + dy)


def _choose_cell_size_for_count(poly_m: Polygon, target_cells: int) -> float:
    """
    For a target number of (square) cells inside poly_m, return a side length (meters).
    """
    area = float(poly_m.area)
    side = np.sqrt(max(area / max(target_cells, 1), 1.0))
    return side

_GEOD = Geod(ellps="WGS84")

def _dist_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    _, _, d_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return d_m / 1000.0


# ---------------------------
# AOIEnvironment
# ---------------------------

Geom = Union[Polygon, MultiPolygon]


@dataclass
class AOIEnvironment:
    """
    Environment with:
      - AOI polygon (water-only, EPSG:4326)
      - Poisson-disk points inside AOI (lon/lat degrees)
      - Fixed grid partition (EPSG:4326) independent of points
      - Graph over grid cells; node centroids computed in projected CRS
    """
    polygon: Geom
    poisson_radius_deg: float = 0.02
    seed: int = 42

    # Generated / derived
    points: Optional[np.ndarray] = None                 # shape (N, 2) in (lon, lat)
    partition_gdf: Optional[gpd.GeoDataFrame] = None    # grid cells in EPSG:4326
    graph: Optional[nx.Graph] = None                    # adjacency over cells
    poisson_graph: Optional[nx.Graph] = None


    # ---------------
    # Constructors
    # ---------------

    @classmethod
    def from_geojson(cls,
                     outer_geojson: str,
                     islands_geojson: str,
                     poisson_radius_deg: float = 0.02,
                     seed: int = 42) -> "AOIEnvironment":
        """
        Build AOIEnvironment from outer boundary and island polygons (both GeoJSON paths).
        Resulting AOI polygon is outer minus islands (water-only).
        """
        outer_gdf = gpd.read_file(outer_geojson)
        islands_gdf = gpd.read_file(islands_geojson)

        outer_union = unary_union(outer_gdf.geometry)
        islands_union = unary_union(islands_gdf.geometry)

        water_polygon = outer_union.difference(islands_union)
        return cls(water_polygon, poisson_radius_deg=poisson_radius_deg, seed=seed)

    # ---------------
    # Poisson sampling
    # ---------------

    def generate_poisson_points(self, include_boundary: bool = True) -> np.ndarray:
        """
        Generate Poisson-disk samples (lon, lat) inside AOI polygon in EPSG:4326.
        Stores in self.points and returns the array.
        """
        minx, miny, maxx, maxy = self.polygon.bounds
        sampler = PoissonDisk(
            d=2,
            radius=self.poisson_radius_deg,
            l_bounds=[minx, miny],
            u_bounds=[maxx, maxy],
            seed=self.seed,
        )
        samples = np.asarray(sampler.fill_space())  # (N,2) in (lon, lat)

        # Keep only inside (and maybe boundary-touching) samples
        def _keep(xy):
            p = Point(xy[0], xy[1])
            # .covers treats boundary as inside; fall back to .touches if include_boundary is False
            return self.polygon.covers(p) if include_boundary else self.polygon.contains(p)

        mask = np.fromiter((bool(_keep(xy)) for xy in samples), count=len(samples), dtype=bool)
        pts = samples[mask]

        self.points = pts
        return pts

    # ---------------
    # Grid partitioning (independent of points)
    # ---------------

    def build_grid_partition(self,
                             cell_width_m: Optional[float] = None,
                             cell_height_m: Optional[float] = None,
                             target_cells: Optional[int] = None,
                             label_boundary: bool = True,
                             ports_gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        Partition AOI into a rectangular grid intersected with the AOI polygon.

        Args:
            cell_width_m:  width of each grid cell in meters (projected CRS).
            cell_height_m: height of each grid cell in meters (defaults to cell_width_m).
            target_cells:  alternatively, choose cell size to achieve ~this many cells.
            label_boundary: add 'is_boundary' flag (cell touches AOI exterior).
            ports_gdf:     optional port polygons (EPSG:4326) to add 'is_port' flag.

        Returns:
            GeoDataFrame (EPSG:4326) with columns: ['cell_id','is_boundary','is_port','geometry']
        """
        # Project AOI to meters
        aoi_gdf = gpd.GeoDataFrame(geometry=[self.polygon], crs="EPSG:4326")
        aoi_m, utm = _to_utm(aoi_gdf)
        poly_m = aoi_m.geometry.iloc[0]

        # Determine cell size
        if target_cells is not None and cell_width_m is None:
            side = _choose_cell_size_for_count(poly_m, target_cells)
            cell_width_m = side
            cell_height_m = side

        if cell_width_m is None:
            raise ValueError("Provide cell_width_m or target_cells.")
        if cell_height_m is None:
            cell_height_m = cell_width_m

        # Build grid in projected coords
        minx, miny, maxx, maxy = poly_m.bounds
        pad = 1e-6  # small pad to reduce edge precision misses
        boxes = list(_grid_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad,
                                  float(cell_width_m), float(cell_height_m)))
        grid_m = gpd.GeoDataFrame({"grid_id": range(len(boxes))}, geometry=boxes, crs=utm)

        # Intersect with AOI
        inter_m = gpd.overlay(grid_m, aoi_m, how="intersection")
        inter_m = inter_m[inter_m.area > 0].reset_index(drop=True)

        # Boundary label (touches AOI exterior)
        if label_boundary:
            aoi_exterior = poly_m.boundary
            inter_m["is_boundary"] = inter_m.geometry.boundary.intersects(aoi_exterior)
        else:
            inter_m["is_boundary"] = False

        # Port label
        if ports_gdf is not None and not ports_gdf.empty:
            ports_m = ports_gdf.to_crs(utm)
            ports_union = ports_m.unary_union
            inter_m["is_port"] = inter_m.geometry.intersects(ports_union)
        else:
            inter_m["is_port"] = False

        # Back to WGS84 + finalize
        out = inter_m.to_crs("EPSG:4326").reset_index(drop=True)
        out["cell_id"] = out.index
        self.partition_gdf = out[["cell_id", "is_boundary", "is_port", "geometry"]]
        return self.partition_gdf

    # ---------------
    # Graph over grid cells (centroids in projected CRS, returned as lon/lat)
    # ---------------

    def build_graph(self) -> nx.Graph:
        """
        Build an adjacency graph over current partition_gdf.
        Nodes: cell_id
        Node attrs: centroid_lon/centroid_lat (computed in UTM then transformed back to WGS84)
        Edges: cells that touch (queen adjacency; corner or edge).
        """
        if self.partition_gdf is None or self.partition_gdf.empty:
            raise ValueError("No partition available. Call build_grid_partition() first.")

        gdf_wgs = self.partition_gdf
        # Project cells to UTM for accurate centroids
        gdf_m, utm = _to_utm(gdf_wgs.copy())
        centroids_m = gdf_m.geometry.centroid
        # Convert centroids back to WGS84
        centroids_wgs = gpd.GeoSeries(centroids_m, crs=utm).to_crs("EPSG:4326")

        # Build graph
        G = nx.Graph()
        for cid, cen in zip(gdf_wgs["cell_id"].tolist(), centroids_wgs.tolist()):
            G.add_node(int(cid), centroid_lon=float(cen.x), centroid_lat=float(cen.y))

        # Spatial index in WGS84 is fine for adjacency (topology only)
        sindex = gdf_wgs.sindex
        for i, geom in enumerate(gdf_wgs.geometry):
            bounds = geom.bounds
            eps = 1e-12
            candidates = list(sindex.intersection((bounds[0]-eps, bounds[1]-eps, bounds[2]+eps, bounds[3]+eps)))
            for j in candidates:
                if i >= j:
                    continue
                if geom.touches(gdf_wgs.geometry.iloc[j]) or geom.intersects(gdf_wgs.geometry.iloc[j]):
                    a = int(gdf_wgs.cell_id.iloc[i])
                    b = int(gdf_wgs.cell_id.iloc[j])
                    if a != b:
                        G.add_edge(a, b)

        self.graph = G
        return G

    def build_poisson_graph(
        self,
        k: int = 12,
        max_edge_km: float = 12.0,
        sample_step_km: float = 2.0,
    ) -> nx.Graph:
        """
        Build a k-NN graph on current Poisson points.
        - Connect each point to up to k nearest neighbors
        - Keep only edges whose geodesic length <= max_edge_km
        - Drop edges whose segment exits the AOI (sampled containment test)
        Stores the result in self.poisson_graph and returns it.
        """
        if self.points is None or len(self.points) == 0:
            self.generate_poisson_points()
    
        pts = self.points  # (N,2) [lon, lat]
        if len(pts) < 2:
            self.poisson_graph = nx.Graph()
            return self.poisson_graph
    
        # KDTree in approx-isotropic lon/lat space
        lat0 = float(np.mean(pts[:, 1]))
        scale = np.cos(np.radians(lat0))
        pts_xy = np.column_stack([pts[:, 0] * scale, pts[:, 1]])
        tree = cKDTree(pts_xy)
    
        G = nx.Graph()
        for i, (lon_i, lat_i) in enumerate(pts):
            G.add_node(i, lon=float(lon_i), lat=float(lat_i))
    
        # Query k+1 because the nearest neighbor is the point itself
        dists, idxs = tree.query(pts_xy, k=min(k + 1, len(pts)))
        # Normalize shapes for k==1 cases
        if np.ndim(idxs) == 1:
            idxs = idxs[:, None]
            dists = dists[:, None]
    
        # Helper: sampled containment test (linear in lon/lat is fine at small steps)
        def segment_inside(lon1, lat1, lon2, lat2) -> bool:
            total_km = _dist_km(lon1, lat1, lon2, lat2)
            if total_km <= sample_step_km:
                # single midpoint check
                t_vals = [0.5]
            else:
                n = max(int(np.ceil(total_km / sample_step_km)), 1)
                # exclude endpoints (guaranteed inside); sample interior
                t_vals = [(i + 1) / (n + 1) for i in range(n)]
            for t in t_vals:
                lon = lon1 + t * (lon2 - lon1)
                lat = lat1 + t * (lat2 - lat1)
                if not self.polygon.covers(Point(lon, lat)):
                    return False
            return True
    
        # Build candidate edges
        for i in range(len(pts)):
            lon_i, lat_i = float(pts[i, 0]), float(pts[i, 1])
            nbrs = idxs[i]
            for j in nbrs:
                j = int(j)
                if j == i:
                    continue
                lon_j, lat_j = float(pts[j, 0]), float(pts[j, 1])
                d_km = _dist_km(lon_i, lat_i, lon_j, lat_j)
                if d_km > float(max_edge_km):
                    continue
                if not segment_inside(lon_i, lat_i, lon_j, lat_j):
                    continue
                # undirected simple graph
                if not G.has_edge(i, j):
                    G.add_edge(i, j, dist_km=float(d_km))
    
        self.poisson_graph = G
        return G
    

    # ---------------
    # Convenience
    # ---------------

    def sample_random_point_in_cell(self, cell_id: int, n_trials: int = 1000, rng: Optional[np.random.Generator] = None) -> tuple[float, float]:
        """
        Rejection-sample a random point inside a grid cell. Returns (lon, lat).
        Uses provided RNG if given; otherwise creates a fresh one.
        """
        if self.partition_gdf is None:
            raise ValueError("No partition. Call build_grid_partition() first.")
        geom = self.partition_gdf.loc[self.partition_gdf["cell_id"] == cell_id, "geometry"].iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        rng = rng or np.random.default_rng()
        for _ in range(max(n_trials, 1)):
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            p = Point(x, y)
            if geom.contains(p):
                return float(x), float(y)
        # fallback to centroid if rejection fails (compute safely via projection)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        gdf_m, utm = _to_utm(gdf)
        cen_m = gdf_m.geometry.centroid
        cen_wgs = gpd.GeoSeries(cen_m, crs=utm).to_crs("EPSG:4326").iloc[0]
        return float(cen_wgs.x), float(cen_wgs.y)

    def build_poisson_index(self):
        """
        Build a GeoDataFrame + spatial index for the current Poisson points.
        Call this after generate_poisson_points() (it auto-calls if needed).
        """
        if self.points is None or len(self.points) == 0:
            self.generate_poisson_points()

        # points is Nx2 array: (lon, lat)
        pts = [Point(lon, lat) for lon, lat in self.points]
        self._points_gdf = gpd.GeoDataFrame(
            {"pid": np.arange(len(pts), dtype=int)}, geometry=pts, crs="EPSG:4326"
        )
        # triggers .sindex on demand; no return needed

    def _ensure_points_index(self):
        if not hasattr(self, "_points_gdf") or self._points_gdf is None:
            self.build_poisson_index()

    def poisson_points_in_cell(self, cell_id: int):
        """
        Return (pid_array, coords_array) of Poisson points that lie *inside* the cell polygon.
        """
        if self.partition_gdf is None:
            raise ValueError("No partition grid. Call build_grid_partition(...) first.")
        self._ensure_points_index()

        row = self.partition_gdf.loc[self.partition_gdf["cell_id"] == cell_id]
        if row.empty:
            return np.array([], dtype=int), np.empty((0, 2))

        cell_geom = row.geometry.values[0]

        # Fast prefilter via bbox then precise interior test (.covers to include boundary)
        cand_idx = list(self._points_gdf.sindex.intersection(cell_geom.bounds))
        if not cand_idx:
            return np.array([], dtype=int), np.empty((0, 2))

        subset = self._points_gdf.iloc[cand_idx]
        inside_mask = subset.geometry.apply(cell_geom.covers)  # boundary-inclusive
        inside = subset.loc[inside_mask]

        if inside.empty:
            return np.array([], dtype=int), np.empty((0, 2))

        pids = inside["pid"].to_numpy()
        coords = np.column_stack([inside.geometry.x.to_numpy(), inside.geometry.y.to_numpy()])
        return pids, coords

    def sample_poisson_point_in_cell(self, cell_id: int, rng: Optional[np.random.Generator] = None):
        """
        Pick a random *Poisson* point inside the cell.
        Fallback: cell centroid if the cell has no Poisson points.
        """
        rng = rng or np.random.default_rng()
        pids, coords = self.poisson_points_in_cell(cell_id)
        if coords.shape[0] == 0:
            # fallback to centroid (rare for very small cells vs radius)
            poly = self.partition_gdf.loc[self.partition_gdf["cell_id"] == cell_id].geometry.values[0]
            c = poly.centroid
            return float(c.x), float(c.y)
        idx = int(rng.integers(0, coords.shape[0]))
        return float(coords[idx, 0]), float(coords[idx, 1])
