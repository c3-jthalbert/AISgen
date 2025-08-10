# aisgen/plot_utils.py
# aisgen/plot_utils.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .geojson_utils import plot_geojson_polygon, polygon_to_feature_collection, gdf_to_feature_collection

def init_map(center_lat=23.0, center_lon=120.0, zoom=6):
    fig = go.Figure()
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center={"lat": center_lat, "lon": center_lon}, zoom=zoom),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    return fig

def add_aoi_outline_from_polygon(aoi_polygon, fig=None, color="green", zoom=7, name="AOI"):
    gj = polygon_to_feature_collection(aoi_polygon, props={"name": name})
    return plot_geojson_polygon(gj, fig=fig, color=color, zoom=zoom)


def add_aoi_outline(aoi_polygon, fig=None, color="green", name="AOI"):
    """
    Add AOI polygon outline(s) to a Plotly map.
    aoi_polygon: shapely Polygon or MultiPolygon in WGS84 (lon/lat).
    """
    if fig is None:
        fig = init_map()
    geoms = [aoi_polygon] if aoi_polygon.geom_type == "Polygon" else list(aoi_polygon.geoms)
    for geom in geoms:
        xs, ys = geom.exterior.coords.xy
        fig.add_trace(go.Scattermapbox(
            lon=list(xs), lat=list(ys),
            mode="lines",
            line=dict(width=2, color=color),
            name=name,
            hoverinfo="skip",
            showlegend=True
        ))
    return fig

def add_poisson_points(points_lonlat, fig=None, name="Poisson Samples", size=5, color="blue"):
    """
    Plot Poisson-disk samples on the map.

    points_lonlat: numpy array shape (N, 2) as (lon, lat)
    """
    if fig is None:
        # center roughly on mean of points if provided
        center_lon = float(points_lonlat[:,0].mean()) if len(points_lonlat) else 120.0
        center_lat = float(points_lonlat[:,1].mean()) if len(points_lonlat) else 23.0
        fig = init_map(center_lat=center_lat, center_lon=center_lon, zoom=6)

    df = pd.DataFrame(points_lonlat, columns=["Longitude", "Latitude"])
    scatter = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
    )
    scatter.update_traces(
        marker=dict(size=size, color=color),
        name=name,
        hoverinfo="skip",
        showlegend=True
    )
    for tr in scatter.data:
        fig.add_trace(tr)
    return fig

def add_grid_cells(partition_gdf, fig=None, line_color="orange", line_width=1, name="Grid Cells"):
    """
    Draw grid cells (GeoDataFrame in WGS84) as outlines.
    """
    if fig is None:
        # try to center on AOI/grid extent
        center = partition_gdf.geometry.unary_union.centroid
        fig = init_map(center_lat=center.y, center_lon=center.x, zoom=6)

    # GeoPandas explore would be easier, but keep it consistent with mapbox lines
    for _, row in partition_gdf.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue
        # handle MultiPolygon
        geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for g in geoms:
            xs, ys = g.exterior.coords.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(xs), lat=list(ys),
                mode="lines",
                line=dict(width=line_width, color=line_color),
                name=name,
                hoverinfo="skip",
                showlegend=False  # avoid legend spam
            ))
    return fig

# --- append to aisgen/plot_utils.py ---


def fit_map_to_bounds(fig, bounds, pad_deg=0.2, max_zoom=12):
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx+maxx)/2, (miny+maxy)/2
    span = max(maxx-minx, maxy-miny) + pad_deg
    zoom = 3 if span>40 else 4 if span>20 else 5 if span>10 else 6 if span>5 else 7 if span>2.5 else 8 if span>1.2 else 9 if span>0.6 else 10 if span>0.3 else 11
    zoom = min(zoom, max_zoom)
    fig.update_layout(mapbox=dict(center={"lat": cy, "lon": cx}, zoom=zoom))
    return fig

def add_grid_cells_outline(partition_gdf, fig=None, color="orange", zoom=None, name="Grid Cells"):
    """
    Draw grid cells as outlines by reusing the proven plot_geojson_polygon path.
    Good up to a few hundred cells.
    """
    gj = gdf_to_feature_collection(partition_gdf)
    fig = plot_geojson_polygon(gj, fig=fig, color=color, zoom=(zoom or 6))
    return fig

def add_grid_cells_filled(partition_gdf, fig=None, line_color="#fa0", line_width=1, fill_opacity=0.12, name="Grid Cells"):
    """
    Alternative: add as a single Mapbox layer (fill + outline). No per-cell colors.
    """
    import plotly.graph_objects as go

    if fig is None:
        fig = init_map()

    gj = gdf_to_feature_collection(partition_gdf)
    minx, miny, maxx, maxy = partition_gdf.total_bounds
    fig = fit_map_to_bounds(fig, (minx, miny, maxx, maxy))

    layers = [
        dict(sourcetype="geojson", source=gj, type="fill", color=line_color, opacity=fill_opacity),
        dict(sourcetype="geojson", source=gj, type="line", color=line_color, line={"width": line_width}),
    ]
    # Merge with any existing layers
    mb = fig.layout.get("mapbox", {})
    existing = list(mb.get("layers", []))
    fig.update_layout(mapbox=dict(layers=existing + layers))
    return fig

def add_boundary_cells(partition_gdf, fig=None, color="#d33", width=2, name="Boundary Cells"):
    """
    Emphasize boundary cells' outlines on top of the grid.
    """
    if "is_boundary" not in partition_gdf.columns:
        return fig
    subset = partition_gdf.loc[partition_gdf["is_boundary"]]
    if subset.empty:
        return fig
    return add_grid_cells_outline(subset, fig=fig, color=color, name=name)

