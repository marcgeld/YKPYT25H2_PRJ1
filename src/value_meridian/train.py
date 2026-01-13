from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ValueMeridian price model.")
    p.add_argument("--csv", required=True, help="Input CSV (EDA-ready) path")
    p.add_argument("--target", required=True, help="Target column name, e.g. sold_price_raw")
    p.add_argument("--model-out", required=True, help="Output model .joblib path")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--market-index-out", default="data/market_index.csv")
    return p.parse_args()


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def median_ape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.median(ape))


def ensure_sqm_price(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Ensure sold_sqm_price_raw exists if possible.
    If missing, compute from sold_price_raw / living_area_raw when both exist.
    """
    if "sold_sqm_price_raw" in df.columns:
        return df

    if target != "sold_price_raw":
        # Only compute this when target is sold_price_raw; otherwise it may not make sense.
        return df

    if "living_area_raw" not in df.columns:
        return df

    with np.errstate(divide="ignore", invalid="ignore"):
        sqm = df["sold_price_raw"] / df["living_area_raw"]
    df = df.copy()
    df["sold_sqm_price_raw"] = sqm
    return df


def build_market_index(df: pd.DataFrame, year_col="sold_year", month_col="sold_month",
                       sqm_price_col="sold_sqm_price_raw") -> pd.DataFrame:
    """
    Market index built from monthly median sqm price.
    Output factor is normalized so the LAST available month has factor=1.0,
    and earlier months have factor = last_median / month_median.
    That lets you adjust older base prices to "today-ish" by multiplying.
    """
    required = {year_col, month_col, sqm_price_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Cannot build market index. Need columns: {year_col}, {month_col}, {sqm_price_col}")

    tmp = df[[year_col, month_col, sqm_price_col]].dropna().copy()
    tmp[year_col] = tmp[year_col].astype(int)
    tmp[month_col] = tmp[month_col].astype(int)

    g = (
        tmp.groupby([year_col, month_col])[sqm_price_col]
        .median()
        .reset_index()
        .rename(columns={sqm_price_col: "median_sqm_price"})
        .sort_values([year_col, month_col])
    )
    if g.empty:
        raise ValueError("Market index grouping produced no rows.")

    last = float(g["median_sqm_price"].iloc[-1])
    g["factor_to_last"] = last / g["median_sqm_price"]
    return g


def make_preprocessor(feature_cols: List[str]) -> Tuple[ColumnTransformer, List[str]]:
    """
    Everything is numeric already. We'll median-impute + scale.
    """
    numeric_features = feature_cols
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric_features


def train_and_eval(
        name: str,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_cols: List[str],
        transform: Optional[str] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    pre, _ = make_preprocessor(feature_cols)

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    y_train_fit = y_train
    y_test_eval = y_test

    if transform == "log1p":
        y_train_fit = np.log1p(y_train)
        # model predicts log space; we inverse later

    pipe.fit(X_train, y_train_fit)

    pred = pipe.predict(X_test)
    if transform == "log1p":
        pred = np.expm1(pred)

    metrics = {
        "rmse": rmse(y_test_eval, pred),
        "mae": float(mean_absolute_error(y_test_eval, pred)),
        "r2": float(r2_score(y_test_eval, pred)),
        "mape": mape(y_test_eval, pred),
        "smape": smape(y_test_eval, pred),
        "median_ape": median_ape(y_test_eval, pred),
    }

    return pipe, {"model": name, "metrics": metrics, "transform": transform or "none"}


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    model_out = Path(args.model_out)
    meta_out = Path(str(model_out) + ".meta.json")
    market_index_out = Path(args.market_index_out)

    df = pd.read_csv(csv_path)
    df = ensure_sqm_price(df, target=args.target)

    if args.target not in df.columns:
        raise ValueError(f"Target column not found: {args.target}")

    # Basic cleanup: drop rows missing target
    df = df.dropna(subset=[args.target]).copy()

    # Feature columns: everything except target
    feature_cols = [c for c in df.columns if c != args.target]

    # Split
    X = df[feature_cols]
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(args.test_size), random_state=int(args.random_state)
    )

    # Candidate models
    candidates = []

    # Ridge baseline
    candidates.append(("ridge", Ridge(alpha=1.0)))

    # Ridge + log1p
    candidates.append(("ridge_log", Ridge(alpha=1.0)))

    # HistGradientBoosting + log1p (often strong)
    hgb = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=600,
        random_state=int(args.random_state),
    )
    candidates.append(("hgb_log", hgb))

    results: List[Dict[str, Any]] = []
    best_pipe: Optional[Pipeline] = None
    best_name: Optional[str] = None
    best_transform: str = "none"
    best_metrics: Optional[Dict[str, Any]] = None

    for name, est in candidates:
        transform = "log1p" if name.endswith("_log") else "none"
        pipe, info = train_and_eval(
            name=name,
            model=est,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_cols=feature_cols,
            transform=transform if transform != "none" else None,
        )
        results.append(
            {
                "model": name,
                "tuned": False,
                **info["metrics"],
                "params": getattr(est, "get_params", lambda: {})(),
            }
        )

        # choose by RMSE
        if best_metrics is None or info["metrics"]["rmse"] < best_metrics["rmse"]:
            best_pipe = pipe
            best_name = name
            best_transform = info["transform"]
            best_metrics = info["metrics"]

    assert best_pipe is not None and best_name is not None and best_metrics is not None

    # Market index (optional; we build it if possible)
    market_index_path = str(market_index_out)
    try:
        mi = build_market_index(df)
        market_index_out.parent.mkdir(parents=True, exist_ok=True)
        mi.to_csv(market_index_out, index=False)
    except Exception:
        # If it cannot be built, leave it empty path
        market_index_path = ""

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, model_out)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "target": args.target,
        "chosen_model": best_name,
        "target_transform": best_transform,  # "log1p" or "none"
        "metrics": best_metrics,
        "feature_columns": feature_cols,
        "market_index_path": market_index_path,
        "model_path": str(model_out),
        "notes": "Model predicts in SEK after applying inverse transform in inference if needed.",
        "model_params": results[[r["model"] for r in results].index(best_name)].get("params", {}),
    }

    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Trained model: {best_name} (transform={best_transform})")
    print(f"Metrics: {best_metrics}")
    print(f"Using {len(feature_cols)} feature columns from meta.json.")
    print(f"Wrote model: {model_out}")
    print(f"Wrote meta : {meta_out}")
    if market_index_path:
        print(f"Wrote market index: {market_index_out}")

    # Optional: print the comparison table (CSV-ish)
    res_df = pd.DataFrame(results)
    # Keep a stable column order if present
    cols = ["model", "tuned", "rmse", "mae", "r2", "mape", "smape", "median_ape", "params"]
    cols = [c for c in cols if c in res_df.columns]
    print(res_df[cols].to_csv(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
