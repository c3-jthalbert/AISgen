# AISgen — AIS-like Vessel Track Generator

## Overview
AISgen produces **plausible AIS-like vessel tracks** within a user-defined **water-only Area of Interest (AOI)**.  
It uses a **two-stage routing pipeline** to balance **global plausibility** and **local maneuver realism**:

1. **Coarse path** — Fast, topology-aware shortest path on a rectangular grid.  
2. **Refined path** — Turn-limited routing on a Poisson-disk graph inside a buffered corridor around the coarse path.

The authoritative output is a **per-point track table** (CSV or Parquet) containing **full vessel metadata, routing configuration, and kinematics**.  
Minimal-property **GeoJSON LineStrings** can also be emitted for mapping and visualization.

---

## Key Features

### Two-Stage Routing
- **Stage 1 — Grid Pathfinding**  
  Rectangular grid in local UTM, port/boundary preference, shortest-path search.  
  Downsampling to remove backtracks and zig-zags.
- **Stage 2 — Poisson Refinement**  
  Turn-limited A* on a Poisson-disk subgraph within a corridor around the coarse path.  
  Enforces `max_turn_deg_per_s` from vessel kinematics.  
  Builds composite straight-plus-arc primitives for smooth G¹-continuous paths.

### Vessel Templates
- Loaded from YAML with:
  - **Metadata:** IDs, dimensions, owners, etc.  
  - **Kinematics:** speeds, acceleration limits, max turn rate.  
  - **Emitter profile** (optional).
- Random sampling or user-specified values.
  
### Configurable Environment
- AOI polygon (WGS84).  
- Grid parameters: target cell count or width.  
- Poisson graph parameters: radius, `k`-nearest neighbors, max edge length.  
- Corridor width, heading bin count, straightness penalty.

### Deterministic Reproducibility
- All stochastic steps seeded by `rng_seed`.

---

## Outputs

### Per-Point Track Table (authoritative)
Includes one row per sampled time step:
- `TrackID`, `Timestamp` (UTC)  
- `Longitude`, `Latitude` (deg)  
- `Speed_knots`, `Heading_deg`, `TurnRate_degps`  
- `SegmentIndex` (primitive index)  
- `Stage` (e.g., `"sampled"`)  
- All vessel metadata (from template)  
- All routing/config fields (environment + parameters)  

## Example Usage

```python
from aisgen.environment import AOIEnvironment
from aisgen.vessel import VesselTemplate
from aisgen.tracks import TrackGenerator
from aisgen.geojson_utils import load_geojson
from aisgen.vessel import load_vessel_templates

# Initialize environment
env = AOIEnvironment.from_geojson(
    "strait_detail.geojson",
    "magong_islands.geojson",
    poisson_radius_deg=0.02,
    seed=42,
)
cells = env.build_grid_partition(target_cells=60)  # or: cell_width_m=10_000
_ = env.build_graph()
tg = TrackGenerator(env)
_ = env.generate_poisson_points()
env.build_poisson_graph()

# Load all templates from the YAML file
templates = load_vessel_templates("AISgen/data/vessel_class_templates.yaml")
template_list = list(templates.values())

# various methods of creating tracks

vt = templates["Tanker"]  # Pick a vessel class

# Random track from template
geo1, df1 = tg.random_track(vt,track_id="random")

# Random track but same exact metadata/kinematics as another vessel
vessel_instance = vt.sample()  # Fix metadata/kinematics
geo_a, df_a = tg._random_track_instance(vessel_instance,track_id="tracka")
geo_b, df_b = tg._random_track_instance(vessel_instance,track_id="trackb")

# From GeoJSON (external file, e.g. exported from geojson.io)
user_geojson = json.load(open("sample.geojson"))["features"][0]  # extract first feature
geo2, df2 = tg.from_geojson(vt, user_geojson,track_id="from_geojson",refine=True)

# From GeoJSON drawn interactively on Folium map
# (Assume `last_geojson` is captured from a Draw plugin callback)
geo3, df3 = tg.from_geojson(vt, last_geojson,track_id="fromdraw",refine=True)


```
