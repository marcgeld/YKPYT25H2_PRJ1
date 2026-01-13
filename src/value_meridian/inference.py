from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

from .features import POI, compute_features_for_objects
from .schema import load_objects_from_yaml, load_pois_from_yaml

DEFAULT_POI_PATH = "data/reference_points.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference using a trained ValueMeridian model.")
    p.add_argument("--model", required=True, help="Path to model .joblib")
    p.add_argument("--input", required=True, help="YAML file with objects")
    p.add_argument("--poi", default=DEFAULT_POI_PATH, help="POI YAML path (default: data/reference_points.yaml)")
    p.add_argument("--market-index", default=None, help="Optional market index CSV path (overrides meta)")
    p.add_argument("--asof", default=None, help="YYYY-MM to represent 'sell today' date for time features")
    p.add_argument("--json", action="store_true", help="Print JSON only (no meta text)")
    return p.parse_args()


def read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_model_meta(model_path: Path) -> Dict[str, Any]:
    meta_path = Path(str(model_path) + ".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def load_market_index(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # expected columns: sold_year, sold_month, median_sqm_price, factor_to_last
    return df


def parse_asof(asof: Optional[str]) -> Tuple[int, int]:
    if asof is None:
        now = datetime.now()
        return int(now.year), int(now.month)
    parts = asof.split("-")
    if len(parts) != 2:
        raise ValueError("--asof must be YYYY-MM")
    return int(parts[0]), int(parts[1])


def infer_transform_inverse(pred: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(pred)
    return pred


def market_factor_for_asof(market_index: pd.DataFrame, asof_year: int, asof_month: int) -> Tuple[float, Dict[str, Any]]:
    """
    Returns factor_to_last for (asof_year, asof_month) if exists.
    Otherwise fallback to last available month factor (which is 1.0 by construction).
    """
    if market_index is None or market_index.empty:
        return 1.0, {"mode": "none"}

    # normalize column names if needed
    cols = set(market_index.columns)
    if {"sold_year", "sold_month"}.issubset(cols) is False:
        # maybe the file uses year/month
        if {"year", "month"}.issubset(cols):
            mi = market_index.rename(columns={"year": "sold_year", "month": "sold_month"}).copy()
        else:
            return 1.0, {"mode": "none"}
    else:
        mi = market_index.copy()

    if "factor_to_last" not in mi.columns:
        return 1.0, {"mode": "none"}

    row = mi[(mi["sold_year"] == asof_year) & (mi["sold_month"] == asof_month)]
    if len(row) == 1:
        return float(row["factor_to_last"].iloc[0]), {"mode": "exact", "year": asof_year, "month": asof_month}

    # fallback: last available
    last = mi.sort_values(["sold_year", "sold_month"]).iloc[-1]
    return float(last["factor_to_last"]), {"mode": "fallback_last", "year": int(last["sold_year"]),
                                           "month": int(last["sold_month"])}


def ensure_time_fields_if_needed(features: Dict[str, float], feature_columns: List[str], asof_year: int,
                                 asof_month: int) -> None:
    """
    If model expects sold_year/sold_month, ensure they exist in features.
    """
    if "sold_year" in feature_columns and "sold_year" not in features:
        features["sold_year"] = float(asof_year)
    if "sold_month" in feature_columns and "sold_month" not in features:
        features["sold_month"] = float(asof_month)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    poi_path = Path(args.poi)

    pipe = joblib.load(model_path)
    meta = load_model_meta(model_path)

    feature_columns: List[str] = meta.get("feature_columns") or []
    target_transform: str = meta.get("target_transform", "none")

    # ASOF (sell today)
    asof_year, asof_month = parse_asof(args.asof)

    # Load POIs (your schema uses lat/lon)
    poi_cfg = read_yaml(poi_path)
    pois: List[POI] = load_pois_from_yaml(poi_cfg)

    # Load inference objects
    inp_cfg = read_yaml(input_path)
    objects = load_objects_from_yaml(inp_cfg)

    # Market index (only used if you want an additional index scaling step)
    market_index_path = args.market_index or meta.get("market_index_path") or ""
    market_index = load_market_index(Path(market_index_path)) if market_index_path else None
    market_factor, market_factor_info = market_factor_for_asof(market_index, asof_year, asof_month)

    # If model includes sold_year/month as features, it already models time.
    # In that case, we SHOULD NOT apply market_factor again (double counts).
    model_is_time_aware = ("sold_year" in feature_columns) or ("sold_month" in feature_columns)

    out_predictions: List[Dict[str, Any]] = []

    for home in objects:
        feats, dists = compute_features_for_objects(home, pois, decimals=1)

        # Ensure time features if model needs them
        if model_is_time_aware:
            ensure_time_fields_if_needed(feats, feature_columns, asof_year, asof_month)

        # Build dataframe in correct column order; fill missing features with NaN
        if not feature_columns:
            # fallback: use keys present (not recommended)
            feature_columns = sorted(feats.keys())

        row = {c: feats.get(c, float("nan")) for c in feature_columns}
        X = pd.DataFrame([row], columns=feature_columns)

        pred = pipe.predict(X)
        pred = infer_transform_inverse(pred, target_transform)
        base_price = float(pred[0])

        if model_is_time_aware:
            # Model already conditions on asof time features, so no extra scaling.
            estimate_today = base_price
            used_factor = 1.0
            used_info = {"mode": "not_applied_time_aware", "year": asof_year, "month": asof_month}
        else:
            # Timeless model can be scaled by index
            estimate_today = base_price * float(market_factor)
            used_factor = float(market_factor)
            used_info = market_factor_info

        living_area = float(home.get("living_area_raw"))
        per_sqm = estimate_today / living_area if living_area and living_area > 0 else None

        out_predictions.append(
            {
                "name": home.get("name", ""),
                "estimate_base_price": base_price,
                "market_factor": used_factor,
                "market_factor_info": used_info,
                "estimate_price_today": estimate_today,
                "estimate_price_today_per_sqm": per_sqm,
                "input": home,
                "features_used": row,
                "distances_km": dists,
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="microseconds"),
        "model_path": str(model_path),
        "poi_path": str(poi_path),
        "market_index_path": market_index_path,
        "asof": {"year": asof_year, "month": asof_month},
        "count": len(out_predictions),
        "predictions": out_predictions,
    }

    if not args.json and meta:
        print("Model meta:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print(f"Using {len(feature_columns)} feature columns from meta.json.")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
