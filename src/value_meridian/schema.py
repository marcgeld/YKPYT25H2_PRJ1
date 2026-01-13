from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .features import POI


def require_key(obj: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise ValueError(f"Missing required field '{key}' in {ctx}")
    return obj[key]


def load_pois_from_yaml(poi_cfg: Mapping[str, Any]) -> List[POI]:
    """
    Expect structure:
      poi:
        - key: partille_center
          name: "Partille centrum"
          category: center
          lat: 57.7396
          lon: 12.1064
    """
    items = poi_cfg.get("poi")
    if not isinstance(items, list):
        raise ValueError("POI config must contain key 'poi' as a list")

    pois: List[POI] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"POI entry #{i} must be a mapping")
        key = str(require_key(item, "key", f"POI#{i}"))
        name = str(item.get("name", key))
        category = str(require_key(item, "category", f"POI {key}"))

        # IMPORTANT: your file uses lat/lon
        lat = float(require_key(item, "lat", f"POI {key}"))
        lon = float(require_key(item, "lon", f"POI {key}"))
        pois.append(POI(key=key, name=name, category=category, lat=lat, lon=lon))

    return pois


def load_objects_from_yaml(input_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Expect:
      objects:
        - name: ...
          type: Villa
          latitude: ...
          longitude: ...
          living_area_raw: ...
          plot_area_raw: ...
          rooms_raw: ...
          operating_cost_raw: ...
          house_age: ...
          sold_year: (optional)
          sold_month: (optional)
    """
    items = input_cfg.get("objects")
    if not isinstance(items, list) or not items:
        raise ValueError("Input YAML must contain 'objects' as a non-empty list")

    out: List[Dict[str, Any]] = []
    for i, obj in enumerate(items):
        if not isinstance(obj, dict):
            raise ValueError(f"Object #{i} must be a mapping")
        # Hard requirements for inference
        for req in ("type", "latitude", "longitude", "living_area_raw"):
            require_key(obj, req, f"object#{i}")

        out.append(dict(obj))
    return out
