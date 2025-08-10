# aisgen/geojson_utils.py

from __future__ import annotations

import json
from typing import Dict, Any, Iterable, Callable, Optional

import pandas as pd
import geopandas as gpd
import plotly.express as px
from shapely.geometry import (
    shape as _shape,
    mapping as _mapping,
    Point,
    Polygon,
    MultiPolygon,
)


# ---------------------------
# I/O
# ---------------------------

def load_geojson(filename: str) -> Dict[str, Any]:
    """Load a GeoJSON file into a Python dict."""
    with open(filename, "r") as f:
        return json.load(f)


# ---------------------------
# Converters
# ---------------------------

def polygon_to_feature_collection(poly: Polygon | MultiPolygon, props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert a shapely Polygon/MultiPolygon (WGS84 lon/lat) to a GeoJSON FeatureCollection.
    """
    props = props or {}
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": props,
            "geometry": _mapping(poly)
        }]
    }


def gdf_to_feature_collection(gdf: gpd.GeoDataFrame, props_fn: Optional[Callable[[pd.Series], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Convert a GeoDataFrame (assumed WGS84) to a single GeoJSON FeatureCollection.
    Optionally attach per-feature properties via props_fn(row) -> dict.
    """
    features: list[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        props = props_fn(row) if props_fn else {}
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": _mapping(row.geometry)
        })
    return {"type": "FeatureCollection", "features": features}


# ---------------------------
# Plotting (Plotly Express *map* APIs)
# ---------------------------

def _features_from_geojson(geojson_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Normalize: return a list of Feature dicts for Polygon/MultiPolygon.
    """
    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data.get("features", [])
    elif geojson_data.get("type") == "Feature":
        features = [geojson_data]
    else:
        # geometry-only object
        features = [{"type": "Feature", "properties": {}, "geometry": geojson_data}]
    return features


def plot_geojson_polygon(
    geojson_data: Dict[str, Any],
    fig=None,
    color: str = "blue",
    zoom: int = 7,
):
    """
    Plot GeoJSON Polygon/MultiPolygon outlines on a map, optionally adding to an existing figure.

    Notes:
    - Uses Plotly Express *map* APIs (px.line_map) per project’s updated style.
    - Accepts FeatureCollection, Feature, or bare geometry objects.
    """
    features = _features_from_geojson(geojson_data)
    if not features:
        raise ValueError("No features found in GeoJSON.")

    all_traces = []
    for feature in features:
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            # coords: [linear_ring, ...]
            for ring in coords:
                df = pd.DataFrame(ring, columns=["lon", "lat"])
                tr = px.line_map(df, lat="lat", lon="lon").data[0]
                tr.line.color = color
                all_traces.append(tr)

        elif gtype == "MultiPolygon":
            # coords: [[linear_ring,...], [linear_ring,...], ...]
            for polygon in coords:
                for ring in polygon:
                    df = pd.DataFrame(ring, columns=["lon", "lat"])
                    tr = px.line_map(df, lat="lat", lon="lon").data[0]
                    tr.line.color = color
                    all_traces.append(tr)

        else:
            raise ValueError(f"Unsupported geometry type '{gtype}'. Only Polygon/MultiPolygon are supported.")

    # Create base fig if needed
    if fig is None:
        fig = px.line_map(pd.DataFrame(), lat=[], lon=[])

    for tr in all_traces:
        fig.add_trace(tr)

    # Auto-center: average of all coordinates (cheap & cheerful)
    all_coords: list[tuple[float, float]] = [c for tr in all_traces for c in zip(tr.lon, tr.lat)]
    if all_coords:
        lons, lats = zip(*all_coords)
        lon_c = sum(lons) / len(lons)
        lat_c = sum(lats) / len(lats)
        fig.update_layout(
            mapbox=dict(center={"lat": float(lat_c), "lon": float(lon_c)}, zoom=int(zoom)),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
    return fig
