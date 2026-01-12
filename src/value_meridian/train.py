"""
ValueMeridian - Train a price model from an EDA-prepared CSV.

- Reads CSV (Pandas)
- Builds a simple market index from monthly median SEK/m² (derived if needed)
- Trains 3 candidate models:
    * ridge (plain target)
    * ridge_log (log1p target)
    * hgb_log (HistGradientBoostingRegressor on log1p target)
- Evaluates on a holdout set with multiple metrics
- Selects the best model by RMSE (lowest)
- Exports:
    * model joblib
    * meta json next to model: <model>.meta.json
    * metrics csv (optional): <model>.metrics.csv
    * market index csv: data/market_index.csv (or --market-index-out)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MARKET_INDEX_OUT = Path("data/market_index.csv")


# ----------------------------
# Utilities
# ----------------------------

def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    d2 = d.replace({0: np.nan})
    return n / d2


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelResult:
    model: str
    tuned: bool
    rmse: float
    mae: float
    r2: float
    mape: float
    smape: float
    median_ape: float
    params: Dict[str, Any]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Avoid divide by 0 in percentage metrics
    eps = 1e-9
    ape = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)
    mape = float(np.mean(ape))

    smape = float(
        np.mean(2.0 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), eps))
    )
    median_ape = float(np.median(ape))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "median_ape": median_ape,
    }


# ----------------------------
# Market index
# ----------------------------

def build_market_index(
    df: pd.DataFrame,
    year_col: str = "sold_year",
    month_col: str = "sold_month",
    sqm_price_col: str = "sold_sqm_price_raw",
    sold_price_col: str = "sold_price_raw",
    living_area_col: str = "living_area_raw",
) -> Optional[pd.DataFrame]:
    """
    Build monthly market index from median SEK/m².

    If sold_sqm_price_raw is missing, derive it as sold_price_raw / living_area_raw.

    Returns None if required pieces are missing.
    """
    if year_col not in df.columns or month_col not in df.columns:
        return None

    tmp = df.copy()

    # Derive sqm price if absent
    if sqm_price_col not in tmp.columns:
        if sold_price_col not in tmp.columns or living_area_col not in tmp.columns:
            return None
        tmp[sqm_price_col] = safe_div(
            pd.to_numeric(tmp[sold_price_col], errors="coerce"),
            pd.to_numeric(tmp[living_area_col], errors="coerce"),
        )

    tmp = ensure_numeric(tmp, [year_col, month_col, sqm_price_col, sold_price_col, living_area_col])

    # Basic sanity filters to reduce garbage influence
    if living_area_col in tmp.columns:
        tmp = tmp[tmp[living_area_col] > 0]
    if sold_price_col in tmp.columns:
        tmp = tmp[tmp[sold_price_col] > 0]

    tmp = tmp.dropna(subset=[year_col, month_col, sqm_price_col])
    if tmp.empty:
        return None

    tmp[year_col] = tmp[year_col].astype(int)
    tmp[month_col] = tmp[month_col].astype(int)

    baseline = float(tmp[sqm_price_col].median())

    g = (
        tmp.groupby([year_col, month_col])[sqm_price_col]
        .median()
        .reset_index()
        .rename(columns={year_col: "year", month_col: "month", sqm_price_col: "median_sqm_price"})
    )
    g["index"] = g["median_sqm_price"] / baseline
    g["source"] = "median_sqm_price_derived"
    g = g.sort_values(["year", "month"]).reset_index(drop=True)

    return g[["year", "month", "index", "source", "median_sqm_price"]]


# ----------------------------
# Feature selection
# ----------------------------

def infer_feature_columns(df: pd.DataFrame, target: str, timeless: bool) -> List[str]:
    """
    Decide which columns to use as features.
    Keeps it simple:
    - Use numeric + bool columns
    - Exclude obvious non-features and the target
    - If timeless=True, exclude sold_year/sold_month (so inference can estimate "today")
    """
    exclude = {target}
    # Keep sold_year/month only if NOT timeless
    if timeless:
        exclude |= {"sold_year", "sold_month"}

    # If you keep helper columns, exclude them here
    exclude |= {"sold_date"}  # often removed; safe to exclude if present

    candidates: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue

        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            candidates.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            candidates.append(col)
        else:
            # allow 0/1 encoded strings? no
            continue

    return sorted(candidates)


def prepare_xy(df: pd.DataFrame, target: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)

    # Drop rows where target is missing/non-numeric
    ok = ~np.isnan(y)
    X = X.loc[ok].copy()
    y = y[ok]

    return X, y


# ----------------------------
# Model builders
# ----------------------------

def make_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    """
    Median-impute + scale numeric columns.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def train_ridge(X_train: pd.DataFrame, y_train: np.ndarray, numeric_cols: List[str], alpha: float = 1.0) -> Pipeline:
    pre = make_preprocessor(numeric_cols)
    model = Ridge(alpha=float(alpha), random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    return pipe


def train_ridge_log(
    X_train: pd.DataFrame, y_train: np.ndarray, numeric_cols: List[str], alpha: float = 1.0
) -> Pipeline:
    y_log = np.log1p(y_train)
    pre = make_preprocessor(numeric_cols)
    model = Ridge(alpha=float(alpha), random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_log)
    return pipe


def train_hgb_log(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    max_depth: int = 6,
    learning_rate: float = 0.08,
    max_iter: int = 600,
) -> Pipeline:
    """
    HistGradientBoostingRegressor handles non-linearities well.
    Trained on log1p(target).
    """
    y_log = np.log1p(y_train)
    pre = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # No scaler needed for HGB, but safe to leave out.
        ]
    )
    preprocessor = ColumnTransformer([("num", pre, numeric_cols)], remainder="drop")

    model = HistGradientBoostingRegressor(
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_log)
    return pipe


def predict_in_sek(pipe: Pipeline, X: pd.DataFrame, target_transform: Optional[str]) -> np.ndarray:
    pred = pipe.predict(X)
    if target_transform == "log1p":
        return np.expm1(pred)
    if target_transform == "log":
        return np.exp(pred)
    return pred


# ----------------------------
# Training orchestration
# ----------------------------

def run_training(
    df: pd.DataFrame,
    target: str,
    timeless: bool,
    test_size: float = 0.2,
) -> Tuple[Pipeline, Dict[str, Any], List[ModelResult]]:
    feature_cols = infer_feature_columns(df, target=target, timeless=timeless)
    if not feature_cols:
        raise ValueError("No usable numeric/bool feature columns found.")

    X, y = prepare_xy(df, target=target, feature_cols=feature_cols)

    # Ensure all features are numeric/bool (coerce bool -> float)
    X = X.copy()
    for c in feature_cols:
        if c in X.columns and pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(float)

    # Drop rows where ALL features are NA (rare)
    X = X.dropna(axis=0, how="all")
    y = y[X.index.to_numpy()]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42
    )

    numeric_cols = feature_cols[:]  # all are numeric/bool by construction

    results: List[ModelResult] = []

    # ridge
    ridge = train_ridge(X_train, y_train, numeric_cols=numeric_cols, alpha=1.0)
    y_pred = predict_in_sek(ridge, X_test, target_transform=None)
    m = compute_metrics(y_test, y_pred)
    results.append(ModelResult("ridge", False, **m, params={"alpha": 1.0}))

    # ridge_log (log1p)
    ridge_log = train_ridge_log(X_train, y_train, numeric_cols=numeric_cols, alpha=1.0)
    y_pred = predict_in_sek(ridge_log, X_test, target_transform="log1p")
    m = compute_metrics(y_test, y_pred)
    results.append(ModelResult("ridge_log", False, **m, params={"alpha": 1.0}))

    # hgb_log (log1p)
    hgb_log = train_hgb_log(
        X_train,
        y_train,
        numeric_cols=numeric_cols,
        max_depth=6,
        learning_rate=0.08,
        max_iter=600,
    )
    y_pred = predict_in_sek(hgb_log, X_test, target_transform="log1p")
    m = compute_metrics(y_test, y_pred)
    results.append(ModelResult("hgb_log", False, **m, params={"max_depth": 6, "learning_rate": 0.08, "max_iter": 600}))

    # Choose best by RMSE
    best = min(results, key=lambda r: r.rmse)

    if best.model == "ridge":
        chosen_pipe = ridge
        transform = None
    elif best.model == "ridge_log":
        chosen_pipe = ridge_log
        transform = "log1p"
    else:
        chosen_pipe = hgb_log
        transform = "log1p"

    meta: Dict[str, Any] = {
        "created_at_utc": now_utc_iso(),
        "csv_path": None,
        "target": target,
        "chosen_model": best.model,
        "timeless_model": bool(timeless),
        "target_transform": transform,  # IMPORTANT: used by inference for inverse-transform
        "metrics": {
            "rmse": best.rmse,
            "mae": best.mae,
            "r2": best.r2,
            "mape": best.mape,
            "smape": best.smape,
            "median_ape": best.median_ape,
        },
        "feature_columns": feature_cols,
        "model_params": best.params,
        "notes": "Model predicts in SEK after applying inverse transform in inference.",
    }

    return chosen_pipe, meta, results


# ----------------------------
# CLI / main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ValueMeridian model and export joblib + meta.")
    p.add_argument("--csv", required=True, help="Input CSV (EDA-prepared).")
    p.add_argument("--target", required=True, help="Target column, e.g. sold_price_raw")
    p.add_argument("--model-out", required=True, help="Output model joblib path.")
    p.add_argument("--timeless", action="store_true", help="Exclude sold_year/sold_month from features.")
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction.")
    p.add_argument(
        "--market-index-out",
        default=str(DEFAULT_MARKET_INDEX_OUT),
        help="Output market index CSV path (default: data/market_index.csv).",
    )
    p.add_argument("--write-metrics-csv", action="store_true", help="Write <model>.metrics.csv next to model.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    model_out = Path(args.model_out)
    market_index_out = Path(args.market_index_out)

    df = pd.read_csv(csv_path)
    df = df.copy()

    # Ensure bool columns stored as True/False are OK
    # No-op in most cases.

    pipe, meta, results = run_training(
        df=df,
        target=str(args.target),
        timeless=bool(args.timeless),
        test_size=float(args.test_size),
    )

    # Build market index if possible (optional)
    market_index = build_market_index(df)
    if market_index is not None:
        market_index_out.parent.mkdir(parents=True, exist_ok=True)
        market_index.to_csv(market_index_out, index=False)
        meta["market_index_path"] = str(market_index_out)
    else:
        meta["market_index_path"] = None

    meta["csv_path"] = str(csv_path)

    # Write model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_out)

    # Write meta next to model
    meta_path = model_out.with_suffix(model_out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional metrics CSV
    if args.write_metrics_csv:
        metrics_path = model_out.with_suffix(model_out.suffix + ".metrics.csv")
        mdf = pd.DataFrame([asdict(r) for r in results])
        mdf.to_csv(metrics_path, index=False)

    print(f"Trained model: {meta['chosen_model']} (timeless={meta['timeless_model']}, transform={meta['target_transform']})")
    print("Metrics:", meta["metrics"])
    print(f"Using {len(meta['feature_columns'])} feature columns from meta.json.")
    print(f"Wrote model: {model_out}")
    print(f"Wrote meta : {meta_path}")
    if meta["market_index_path"]:
        print(f"Wrote market index: {meta['market_index_path']}")
    else:
        print("Market index: not created (missing required columns).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
