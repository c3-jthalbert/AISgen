# vessel.py
"""
Vessel template loader and sampler.

- Reads YAML templates like the ones you posted (keys: metadata, kinematics, emitter_profile)
- Supports simple sampling directives inside metadata AND emitter_profile:
    * one_of: [a, b, c]           -> picks a single value
    * uniform_range: [min, max]   -> uniform float in [min, max]
  Any plain scalar or nested structure is passed through as-is.

- Leaves `kinematics` untouched (directly from YAML).
- `emitter_profile` can be a string, a list of strings, or an object with one_of
  that returns either a string or a list of strings. The resolver normalizes and validates.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union
import random
import yaml
import copy
import uuid

Scalar = Union[str, int, float, bool, None]
JSONLike = Union[Scalar, List["JSONLike"], Dict[str, "JSONLike"]]

# ----------------------------
# Helpers for sampling
# ----------------------------

def _is_one_of_spec(obj: Any) -> bool:
    return isinstance(obj, Mapping) and "one_of" in obj and isinstance(obj["one_of"], list)

def _is_uniform_range_spec(obj: Any) -> bool:
    return (
        isinstance(obj, Mapping)
        and "uniform_range" in obj
        and isinstance(obj["uniform_range"], list)
        and len(obj["uniform_range"]) == 2
    )

def _resolve_value(spec: Any, rng: random.Random) -> Any:
    if _is_one_of_spec(spec):
        choices = spec["one_of"]
        if not choices:
            raise ValueError("one_of list is empty.")
        choice = rng.choice(choices)
        return _resolve_value(choice, rng)

    if _is_uniform_range_spec(spec):
        lo, hi = spec["uniform_range"]
        return float(rng.uniform(float(lo), float(hi)))

    if isinstance(spec, list):
        return [_resolve_value(v, rng) for v in spec]

    if isinstance(spec, Mapping):
        return {k: _resolve_value(v, rng) for k, v in spec.items()}

    return spec

def _resolve_emitter_profile(spec: Any, rng: random.Random) -> Union[str, List[str]]:
    val = _resolve_value(spec, rng)

    if isinstance(val, str):
        return val
    if isinstance(val, list):
        if not all(isinstance(x, str) for x in val):
            raise ValueError("Resolved emitter_profile list must contain only strings.")
        return val

    raise ValueError(f"Resolved emitter_profile must be a string or list[str], got {type(val)}")

# ----------------------------
# Data classes
# ----------------------------

@dataclass(frozen=True)
class VesselTemplate:
    name: str                 # template name (YAML key)
    vessel_type: str          # same as name for clarity
    metadata_spec: Dict[str, Any]
    kinematics: Dict[str, Any]
    emitter_profile_spec: Any

    def sample(self, seed: Optional[int] = None) -> "VesselInstance":
        rng = random.Random(seed)

        resolved_meta = _resolve_value(self.metadata_spec, rng)
        if not isinstance(resolved_meta, Mapping):
            raise ValueError(
                f"Resolved metadata for template '{self.name}' is not a mapping: {type(resolved_meta)}"
            )

        kin = copy.deepcopy(self.kinematics) if self.kinematics is not None else {}

        emit = None
        if self.emitter_profile_spec is not None:
            emit = _resolve_emitter_profile(self.emitter_profile_spec, rng)

        return VesselInstance(
            template_name=self.name,
            metadata=dict(resolved_meta),
            kinematics=kin,
            emitter_profile=emit,
            instance_id=str(uuid.uuid4())
        )

    def sample_speed_knots(self, rng: Optional["np.random.Generator"] = None) -> float:
        import numpy as np
        rng = rng or np.random.default_rng()
    
        min_kn = float(self.kinematics.get("min_speed_kn", 5.0))
        max_kn = float(self.kinematics.get("max_speed_kn", max(min_kn + 1.0, 12.0)))
        cruise = self.kinematics.get("cruise_speed_kn", None)
    
        if cruise is not None:
            cruise = float(cruise)
            std = max(0.1 * cruise, 0.2)
            spd = float(rng.normal(loc=cruise, scale=std))
            return float(np.clip(spd, min_kn, max_kn))
    
        if max_kn <= min_kn:
            return min_kn
        return float(rng.uniform(min_kn, max_kn))

@dataclass(frozen=True)
class VesselInstance:
    template_name: str
    metadata: Dict[str, Any]
    kinematics: Dict[str, Any]
    emitter_profile: Union[str, List[str], None]
    instance_id: str

# ----------------------------
# YAML loading
# ----------------------------

def load_vessel_templates(yaml_path: str) -> Dict[str, VesselTemplate]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError(f"Top-level YAML must be a mapping of template names to definitions, got: {type(data)}")

    templates: Dict[str, VesselTemplate] = {}
    for name, block in data.items():
        if not isinstance(block, Mapping):
            raise ValueError(f"Template '{name}' must be a mapping, got: {type(block)}")

        metadata = block.get("metadata", {})
        kinematics = block.get("kinematics", {})
        emitter = block.get("emitter_profile", None)

        if not isinstance(metadata, Mapping):
            raise ValueError(f"'metadata' for template '{name}' must be a mapping.")
        if kinematics is not None and not isinstance(kinematics, Mapping):
            raise ValueError(f"'kinematics' for template '{name}' must be a mapping or omitted.")
        if emitter is not None and not isinstance(emitter, (str, list, Mapping)):
            raise ValueError(f"'emitter_profile' for template '{name}' must be a string, list, mapping, or omitted.")

        templates[name] = VesselTemplate(
            name=name,
            vessel_type=name,  # now every template has vessel_type
            metadata_spec=dict(metadata),
            kinematics=dict(kinematics) if kinematics is not None else {},
            emitter_profile_spec=emitter
        )

    return templates

# ----------------------------
# Convenience utilities
# ----------------------------

def sample_all(templates: Dict[str, VesselTemplate], seed: Optional[int] = None) -> Dict[str, VesselInstance]:
    out: Dict[str, VesselInstance] = {}
    base = random.Random(seed)
    for name, tmpl in templates.items():
        out[name] = tmpl.sample(seed=base.randrange(1_000_000_000))
    return out


