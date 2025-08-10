# AISgen — AIS-like Vessel Track Generator

## Overview
AISgen generates plausible AIS-like vessel tracks inside a water-only Area of Interest (AOI) using a two-stage routing pipeline:

1. **Coarse path** — Fast, topology-aware shortest path on a rectangular grid.  
2. **Refined path** — Turn-limited routing on a Poisson-disk graph inside a buffered corridor around the coarse path.

The output is a **per-point track table** (CSV or Parquet) carrying all vessel metadata, routing configuration, and kinematics.  
Minimal-property GeoJSONs are optionally emitted for visualization.

---

## Features

### Two-stage routing
- **Stage 1:** Grid-based coarse path with port/boundary preference.  
- **Stage 2:** Poisson-disk refinement constrained by vessel max turn rate.

### Configurable environment
- AOI polygon, grid size or cell count, Poisson radius, corridor width, k-nearest neighbors, and max edge length.

### Vessel templates
- Metadata (IDs, dimensions), kinematics (speeds, turn rate), and optional emitter profile, loaded from YAML.

### Outputs
- **Per-point track table** with:
  - `TrackID`, `Timestamp` (UTC)  
  - `Longitude`, `Latitude` (deg)  
  - `Speed_knots`, `Heading_deg`, `TurnRate_degps`  
  - `SegmentIndex` (currently `0` for all rows in straight-segment mode)  
  - `Stage` (`"sampled"`)  
  - Vessel metadata, routing/config fields
- **Minimal GeoJSON** of refined path (for visualization only).
- **Deterministic** with `rng_seed`.

