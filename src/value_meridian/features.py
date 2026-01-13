from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class POI:
    key: str
    name: str
    category: str
    lat: float
    lon: float


def haversine_vec(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    """Vectorized haversine distance (km) from arrays of lat/lon to a single reference point."""
    R = 6371.0
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = np.radians(float(ref_lat))
    lon2 = np.radians(float(ref_lon))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def distance_feature(lat_series, lon_series, ref_lat: float, ref_lon: float, decimals: int = 1):
    """Return rounded km distance series."""
    arr = haversine_vec(np.asarray(lat_series), np.asarray(lon_series), ref_lat, ref_lon)
    return np.round(arr, decimals)


def compute_distances_for_point(lat: float, lon: float, pois: List[POI], decimals: int = 1) -> Dict[str, float]:
    """Compute dist_<poi.key>_km for a single lat/lon."""
    distances: Dict[str, float] = {}
    for poi in pois:
        d = haversine_vec(np.array([lat]), np.array([lon]), poi.lat, poi.lon)[0]
        distances[f"dist_{poi.key}_km"] = float(np.round(d, decimals))
    return distances


def aggregate_min_distances(distances_km: Mapping[str, float], pois: List[POI]) -> Dict[str, float]:
    """
    Build aggregated min_dist_<category>_km from individual dist_<key>_km.

    Example categories you use: center, transport, workplace, infrastructure, recreation.
    """
    by_cat: Dict[str, List[float]] = {}
    poi_by_key = {p.key: p for p in pois}

    for col, val in distances_km.items():
        if not col.startswith("dist_") or not col.endswith("_km"):
            continue
        key = col[len("dist_"): -len("_km")]
        poi = poi_by_key.get(key)
        if poi is None:
            continue
        by_cat.setdefault(poi.category, []).append(float(val))

    out: Dict[str, float] = {}
    for cat, vals in by_cat.items():
        out[f"min_dist_{cat}_km"] = float(np.min(vals)) if vals else float("nan")

    return out


def type_onehot(type_value: str) -> Dict[str, float]:
    """
    Convert 'type' (Villa/Radhus/Parhus) to one-hot columns.
    """
    t = (type_value or "").strip().lower()
    allowed = {"villa", "radhus", "parhus"}
    if t not in allowed:
        raise ValueError(f"Unsupported type: {type_value!r}. Expected one of: Villa, Radhus, Parhus")

    return {
        "type_Villa": 1.0 if t == "villa" else 0.0,
        "type_Radhus": 1.0 if t == "radhus" else 0.0,
        "type_Parhus": 1.0 if t == "parhus" else 0.0,
    }


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def compute_core_features(home: Mapping[str, Any]) -> Dict[str, float]:
    """
    Compute the "core" engineered features independent of POIs.
    Returns numeric feature dict (floats).
    """
    living_area = safe_float(home.get("living_area_raw"))
    plot_area = safe_float(home.get("plot_area_raw"))
    rooms = safe_float(home.get("rooms_raw"))
    operating_cost = safe_float(home.get("operating_cost_raw"))
    house_age = safe_float(home.get("house_age"))

    rooms_missing = 1.0 if rooms is None else 0.0
    rooms_value = 0.0 if rooms is None else float(rooms)

    plot_living_ratio = None
    if living_area and living_area > 0 and plot_area is not None:
        plot_living_ratio = float(plot_area) / float(living_area)

    feats: Dict[str, float] = {
        "living_area_raw": float(living_area) if living_area is not None else float("nan"),
        "plot_area_raw": float(plot_area) if plot_area is not None else float("nan"),
        "rooms_raw": float(rooms_value),
        "rooms_missing": float(rooms_missing),
        "operating_cost_raw": float(operating_cost) if operating_cost is not None else float("nan"),
        "house_age": float(house_age) if house_age is not None else float("nan"),
        "plot_living_ratio": float(plot_living_ratio) if plot_living_ratio is not None else float("nan"),
    }
    return feats


def compute_features_for_objects(
        home: Mapping[str, Any],
        pois: List[POI],
        decimals: int = 1,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute all model features + all dist_ columns for one home.
    Returns (features_used, distances_km).
    """
    lat = safe_float(home.get("latitude"))
    lon = safe_float(home.get("longitude"))
    if lat is None or lon is None:
        raise ValueError("Missing latitude/longitude in input home object")

    distances = compute_distances_for_point(lat, lon, pois, decimals=decimals)
    agg = aggregate_min_distances(distances, pois)

    feats: Dict[str, float] = {}
    feats.update(compute_core_features(home))
    feats["latitude"] = float(lat)
    feats["longitude"] = float(lon)

    # Type one-hot (based on 'type')
    feats.update(type_onehot(str(home.get("type"))))

    # Time features (optional; will be used if model expects them)
    sold_year = safe_float(home.get("sold_year"))
    sold_month = safe_float(home.get("sold_month"))
    if sold_year is not None:
        feats["sold_year"] = float(sold_year)
    if sold_month is not None:
        feats["sold_month"] = float(sold_month)

    # Include individual dist_ columns + aggregated min_dist columns
    feats.update({k: float(v) for k, v in distances.items()})
    feats.update({k: float(v) for k, v in agg.items()})

    return feats, {**distances, **agg}
