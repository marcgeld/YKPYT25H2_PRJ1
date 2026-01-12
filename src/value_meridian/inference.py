"""
ValueMeridian - Inference CLI

- Loads a trained scikit-learn model from joblib (vm-train output).
- Loads optional model metadata from "<model>.meta.json" and prints it.
- Loads POIs from YAML (supports lat/lon OR latitude/longitude).
- Computes distance features from (latitude, longitude) to each POI.
- Computes aggregated min-dist features per POI category:
    - min_dist_center_km
    - min_dist_transport_km
    - min_dist_workplace_km
    - min_dist_infrastructure_km
    - min_dist_recreation_km
- Accepts YAML input with an array of objects under "objects:".
- If a market index CSV is available, applies a market factor to estimate "price today".

Example:
  uv run vm-infer --model data/partille_model.joblib --input data/example_infer.yaml

Input YAML format:
  objects:
    - name: "Villa i Sävedalen"
      type: Villa
      latitude: 57.7238
      longitude: 12.0630
      living_area_raw: 99
      plot_area_raw: 1040
      rooms_raw: 4
      operating_cost_raw: 3400
      house_age: 59
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import yaml


# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class Poi:
    key: str
    name: str
    category: str
    latitude: float
    longitude: float


# ---------------------------
# Small helpers
# ---------------------------

def normalize_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def require_key(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required field '{key}' in {ctx}")
    return d[key]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------
# Haversine distance
# ---------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two points."""
    r = 6371.0
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    lat2r = math.radians(lat2)
    lon2r = math.radians(lon2)

    dlat = lat2r - lat1r
    dlon = lon2r - lon1r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def round_km(x: float, decimals: int = 1) -> float:
    return round(float(x), decimals)


# ---------------------------
# POI loading
# ---------------------------

def parse_poi_lat_lon(item: Dict[str, Any], ctx: str) -> Tuple[float, float]:
    """
    Accept multiple POI coordinate formats:
      - latitude / longitude
      - lat / lon   (your current YAML)
      - coords: [lat, lon]
      - position: { latitude, longitude } or { lat, lon }
    """
    lat = item.get("latitude", item.get("lat"))
    lon = item.get("longitude", item.get("lon"))

    if (lat is None or lon is None) and isinstance(item.get("position"), dict):
        pos = item["position"]
        lat = lat if lat is not None else pos.get("latitude", pos.get("lat"))
        lon = lon if lon is not None else pos.get("longitude", pos.get("lon"))

    if (lat is None or lon is None) and isinstance(item.get("coords"), (list, tuple)) and len(item["coords"]) >= 2:
        lat = lat if lat is not None else item["coords"][0]
        lon = lon if lon is not None else item["coords"][1]

    lat_f = safe_float(lat)
    lon_f = safe_float(lon)
    if lat_f is None or lon_f is None:
        raise ValueError(
            f"Missing/invalid lat/lon in {ctx}. Expected lat+lon (or latitude+longitude). "
            f"Got keys: {list(item.keys())}"
        )
    return float(lat_f), float(lon_f)


def load_pois(poi_path: Path) -> List[Poi]:
    cfg = yaml.safe_load(poi_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "poi" not in cfg:
        raise ValueError(f"POI YAML must contain a top-level 'poi' list. Path: {poi_path}")

    raw = cfg["poi"]
    if not isinstance(raw, list):
        raise ValueError(f"'poi' must be a list. Path: {poi_path}")

    out: List[Poi] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"POI #{i} must be a dict, got: {type(item).__name__}")

        key = normalize_key(str(require_key(item, "key", f"POI #{i}")))
        name = str(item.get("name") or key)
        category = normalize_key(str(item.get("category") or "unknown"))
        lat, lon = parse_poi_lat_lon(item, ctx=f"POI {key}")

        out.append(Poi(key=key, name=name, category=category, latitude=lat, longitude=lon))

    return out


# ---------------------------
# Market index loading
# ---------------------------

def load_market_index(path: Path) -> Dict[Tuple[int, int], float]:
    """
    Loads market index as a mapping (year, month) -> factor

    Expected CSV columns (flexible):
      - year, month, factor
    or:
      - sold_year, sold_month, market_factor
    or:
      - year, month, index

    We treat the numeric column as multiplicative factor.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    # Identify likely columns
    sample = rows[0].keys()
    year_col = "year" if "year" in sample else ("sold_year" if "sold_year" in sample else None)
    month_col = "month" if "month" in sample else ("sold_month" if "sold_month" in sample else None)

    factor_col = None
    for cand in ("factor", "market_factor", "index"):
        if cand in sample:
            factor_col = cand
            break

    if not year_col or not month_col or not factor_col:
        raise ValueError(
            f"Market index CSV must have year+month+factor columns. "
            f"Found columns: {list(sample)} in {path}"
        )

    out: Dict[Tuple[int, int], float] = {}
    for r in rows:
        y = int(float(r[year_col]))
        m = int(float(r[month_col]))
        fval = safe_float(r[factor_col])
        if fval is None:
            continue
        out[(y, m)] = float(fval)

    return out


def pick_market_factor(
    market: Dict[Tuple[int, int], float],
    year: Optional[int],
    month: Optional[int],
) -> Tuple[float, Dict[str, Any]]:
    """
    If exact (year,month) exists -> use it.
    Else fall back to the latest available point (by (year,month)).
    """
    if not market:
        return 1.0, {"mode": "none"}

    if year is not None and month is not None and (year, month) in market:
        return float(market[(year, month)]), {"mode": "exact", "year": year, "month": month}

    # fallback: last available
    y, m = sorted(market.keys())[-1]
    return float(market[(y, m)]), {"mode": "fallback_last", "year": y, "month": m}


# ---------------------------
# Feature computation
# ---------------------------

TYPE_TO_ONEHOT = {
    "villa": "type_Villa",
    "radhus": "type_Radhus",
    "parhus": "type_Parhus",
    # You can expand later (kedjehus etc.) if your training set includes them.
}


CATEGORY_TO_MINCOL = {
    "center": "min_dist_center_km",
    "transport": "min_dist_transport_km",
    "workplace": "min_dist_workplace_km",
    "infrastructure": "min_dist_infrastructure_km",
    "recreation": "min_dist_recreation_km",
}


def compute_distances_for_home(lat: float, lon: float, pois: List[Poi], decimals: int = 1) -> Dict[str, float]:
    """
    Produces:
      dist_<poi.key>_km for each POI.
    """
    d: Dict[str, float] = {}
    for p in pois:
        km = haversine_km(lat, lon, p.latitude, p.longitude)
        d[f"dist_{p.key}_km"] = round_km(km, decimals=decimals)
    return d


def aggregate_min_distances(distances: Dict[str, float], pois: List[Poi]) -> Dict[str, float]:
    """
    Produces:
      min_dist_center_km, min_dist_transport_km, ...
    """
    by_cat: Dict[str, List[float]] = {}
    for p in pois:
        col = f"dist_{p.key}_km"
        if col in distances:
            by_cat.setdefault(p.category, []).append(float(distances[col]))

    out: Dict[str, float] = {}
    for cat, mincol in CATEGORY_TO_MINCOL.items():
        vals = by_cat.get(cat, [])
        out[mincol] = float(min(vals)) if vals else float("nan")

    return out


def compute_features_for_home(home: Dict[str, Any], pois: List[Poi]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - features_used (flat dict)
      - distances_km (dist_* plus min_dist_*)

    Required for meaningful prediction:
      - type
      - latitude, longitude
      - living_area_raw
    """
    ctx = f"object '{home.get('name', '<unnamed>')}'"

    # Accept latitude/longitude; (you can extend if you want to accept lat/lon in objects too)
    lat = safe_float(home.get("latitude"))
    lon = safe_float(home.get("longitude"))
    if lat is None or lon is None:
        raise ValueError(f"Missing latitude/longitude in {ctx}")

    # Base numeric features (may be missing -> NaN; model pipeline should impute where needed)
    living_area = safe_float(home.get("living_area_raw"))
    plot_area = safe_float(home.get("plot_area_raw"))
    rooms = safe_float(home.get("rooms_raw"))
    operating = safe_float(home.get("operating_cost_raw"))
    house_age = safe_float(home.get("house_age"))

    # Sold timing (optional; used only if model is not timeless)
    sold_year = safe_float(home.get("sold_year"))
    sold_month = safe_float(home.get("sold_month"))

    # One-hot type from "type: Villa/Radhus/Parhus"
    type_raw = str(require_key(home, "type", ctx))
    type_key = normalize_key(type_raw)
    if type_key not in TYPE_TO_ONEHOT:
        raise ValueError(
            f"Unsupported type '{type_raw}' in {ctx}. "
            f"Supported: {sorted(TYPE_TO_ONEHOT.keys())}"
        )

    type_cols = {v: 0.0 for v in TYPE_TO_ONEHOT.values()}
    type_cols[TYPE_TO_ONEHOT[type_key]] = 1.0

    # Derived features
    rooms_missing = 1.0 if rooms is None else 0.0

    plot_living_ratio = float("nan")
    if plot_area is not None and living_area is not None and living_area > 0:
        plot_living_ratio = plot_area / living_area

    # Distances
    distances = compute_distances_for_home(float(lat), float(lon), pois, decimals=1)
    agg = aggregate_min_distances(distances, pois)
    distances_all = {**distances, **agg}

    features: Dict[str, Any] = {
        "living_area_raw": living_area,
        "plot_area_raw": plot_area,
        "rooms_raw": rooms,
        "operating_cost_raw": operating,
        "latitude": float(lat),
        "longitude": float(lon),
        "house_age": house_age,
        "plot_living_ratio": plot_living_ratio,
        "rooms_missing": rooms_missing,
        # Optional timing (only relevant if your model uses them)
        "sold_year": sold_year,
        "sold_month": sold_month,
        **type_cols,
        **distances_all,  # includes dist_* and min_dist_*
    }

    return features, distances_all


# ---------------------------
# Model/meta handling
# ---------------------------

def load_meta(model_path: Path) -> Optional[Dict[str, Any]]:
    meta_path = Path(str(model_path) + ".meta.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def inverse_target_transform(pred: Iterable[float], transform: Optional[str]) -> List[float]:
    """
    Supports:
      - None / "" => identity
      - "log1p"   => expm1
    """
    transform = (transform or "").strip().lower()
    out: List[float] = []
    for v in pred:
        fv = float(v)
        if transform == "log1p":
            out.append(math.expm1(fv))
        else:
            out.append(fv)
    return out


# ---------------------------
# Input loading
# ---------------------------

def load_objects_yaml(path: Path) -> List[Dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Input YAML must be a mapping with 'objects:'. Path: {path}")

    objs = cfg.get("objects")
    if not isinstance(objs, list) or not objs:
        raise ValueError(f"Input YAML must contain a non-empty 'objects:' list. Path: {path}")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(objs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{i}] must be a dict, got: {type(item).__name__}")
        out.append(item)
    return out


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ValueMeridian inference on YAML objects.")
    p.add_argument("--model", required=True, help="Path to joblib model, e.g. data/partille_model.joblib")
    p.add_argument("--input", required=True, help="Path to YAML with objects[]")
    p.add_argument("--poi", default=os.environ.get("VM_POI", "data/reference_points.yaml"),
                   help="Path to POI YAML (default: data/reference_points.yaml or env VM_POI)")
    p.add_argument("--market-index", default=os.environ.get("VM_MARKET_INDEX", ""),
                   help="Path to market index CSV (optional). If omitted, uses meta.json path if present.")
    p.add_argument("--json", action="store_true", help="Print raw JSON only (no extra text).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    poi_path = Path(args.poi)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not input_path.exists():
        raise SystemExit(f"Input YAML not found: {input_path}")
    if not poi_path.exists():
        raise SystemExit(f"POI YAML not found: {poi_path}")

    model = joblib.load(model_path)
    meta = load_meta(model_path)

    # Print meta unless --json
    if meta and not args.json:
        print("Model meta:")
        for k, v in meta.items():
            print(f"  {k}: {v}")

    # Feature columns expected
    feature_columns: Optional[List[str]] = None
    if meta and isinstance(meta.get("feature_columns"), list):
        feature_columns = list(meta["feature_columns"])
        if not args.json:
            print(f"Using {len(feature_columns)} feature columns from meta.json.")

    # Determine transform
    target_transform = None
    if meta:
        target_transform = meta.get("target_transform") or meta.get("targetTransform")

    # Load POIs
    pois = load_pois(poi_path)

    # Market index path
    market_index_path: Optional[Path] = None
    if args.market_index.strip():
        market_index_path = Path(args.market_index.strip())
    elif meta and meta.get("market_index_path"):
        market_index_path = Path(str(meta["market_index_path"]))

    market = load_market_index(market_index_path) if market_index_path else {}

    # Load objects
    objects = load_objects_yaml(input_path)

    # Build per-object features
    rows: List[Dict[str, Any]] = []
    dist_debug: List[Dict[str, Any]] = []
    for obj in objects:
        feats, dists = compute_features_for_home(obj, pois)
        rows.append(feats)
        dist_debug.append(dists)

    # If meta has feature columns: build X strictly with those columns
    if feature_columns:
        X = []
        for r in rows:
            X.append({c: r.get(c, float("nan")) for c in feature_columns})
    else:
        # Fall back to using whatever we computed
        X = rows

    # Predict
    raw_pred = model.predict(X)  # type: ignore[attr-defined]
    pred_sek = inverse_target_transform(raw_pred, target_transform)

    # Market factor: pick based on *today* (or sold_year/sold_month if provided per object)
    today = datetime.now().date()
    today_year, today_month = today.year, today.month

    # If the model is timeless, we should not apply market factor by default.
    timeless = bool(meta.get("timeless_model")) if meta else False

    predictions: List[Dict[str, Any]] = []
    for obj, feats, dists, base_price in zip(objects, rows, dist_debug, pred_sek):
        name = str(obj.get("name") or "unnamed")
        living_area = safe_float(obj.get("living_area_raw")) or safe_float(feats.get("living_area_raw"))

        factor = 1.0
        factor_info = {"mode": "none"}
        if (not timeless) and market:
            factor, factor_info = pick_market_factor(market, today_year, today_month)

        price_today = float(base_price) * float(factor)
        price_today_per_sqm = None
        if living_area and living_area > 0:
            price_today_per_sqm = price_today / float(living_area)

        predictions.append(
            {
                "name": name,
                "estimate_base_price": float(base_price),
                "market_factor": float(factor),
                "market_factor_info": factor_info,
                "estimate_price_today": float(price_today),
                "estimate_price_today_per_sqm": float(price_today_per_sqm) if price_today_per_sqm is not None else None,
                "input": obj,
                "features_used": feats if not feature_columns else {k: feats.get(k) for k in feature_columns},
                "distances_km": dists,
            }
        )

    output = {
        "generated_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "poi_path": str(poi_path),
        "market_index_path": str(market_index_path) if market_index_path else None,
        "count": len(predictions),
        "predictions": predictions,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())