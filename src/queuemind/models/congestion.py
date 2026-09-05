"""Emergency Department congestion and multi-horizon load forecasting module.

Implements:
1. Temporal partitioning with configurable embargo buffers to prevent multi-step
   target overlap leakage.
2. Operational baselines:
   - LastValueCongestionBaseline (Persistence)
   - TimeOfDayMedianCongestionBaseline (Historical diurnal profile)
3. Direct multi-horizon ML models:
   - RidgeCongestionModel (Regularized linear)
   - XGBoostCongestionModel (Gradient-boosted decision trees per horizon)
4. CongestionPredictor: Production inference container integrating forecasts,
   split-conformal intervals, and operational bottleneck indicators.
"""

from __future__ import annotations

from datetime import timedelta
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from queuemind.features.bottleneck_features import (
    classify_congestion_state,
    detect_bottleneck_indicators,
)
from queuemind.models.conformal import ConformalIntervalCalibrator

logger = logging.getLogger(__name__)

PROHIBITED_CONGESTION_COLUMNS: frozenset[str] = frozenset(
    {
        "snapshot_time",
        "target_census_30m",
        "target_census_60m",
        "target_census_120m",
        "outtime",
        "disposition",
        "remaining_time_minutes",
        "remaining_time",
    }
)


def temporal_congestion_split(
    df: pd.DataFrame,
    time_col: str = "snapshot_time",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_minutes: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition time-series snapshots chronologically with strict embargo buffers.

    EMBARGO GUARANTEE:
    When forecasting up to H minutes into the future (e.g. 120m), observations
    near the partition boundary have future targets that overlap the subsequent
    period. Inserting an embargo buffer of >= H minutes guarantees:
        max(train[time_col]) + embargo <= min(val[time_col])
        max(val[time_col]) + embargo <= min(test[time_col])

    Args:
        df: Input DataFrame containing snapshots.
        time_col: Timestamp column name to sort and split by.
        train_ratio: Proportion of time range for training.
        val_ratio: Proportion of time range for validation.
        test_ratio: Proportion of time range for test.
        embargo_minutes: Buffer duration in minutes between splits.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If df is empty, ratios are invalid, or time span is too short.
        KeyError: If time_col is missing.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in DataFrame.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0; got {total_ratio}")

    if min(train_ratio, val_ratio, test_ratio) <= 0.0:
        raise ValueError("Split ratios must be strictly positive.")

    sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
    times = pd.to_datetime(sorted_df[time_col])
    t_min = times.min()
    t_max = times.max()
    total_duration = t_max - t_min

    embargo_delta = timedelta(minutes=embargo_minutes)
    required_buffer = 2 * embargo_delta

    if total_duration <= required_buffer:
        raise ValueError(
            f"Total duration ({total_duration}) is too short for split with "
            f"2x {embargo_minutes}m embargoes ({required_buffer})."
        )

    effective_duration = total_duration - required_buffer
    t_train_end = t_min + effective_duration * train_ratio
    t_val_start = t_train_end + embargo_delta
    t_val_end = t_val_start + effective_duration * val_ratio
    t_test_start = t_val_end + embargo_delta

    train_df = sorted_df[times <= t_train_end].copy()
    val_df = sorted_df[(times >= t_val_start) & (times <= t_val_end)].copy()
    test_df = sorted_df[times >= t_test_start].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            f"Dataset produced empty split: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}. Ensure sufficient snapshots."
        )

    return train_df, val_df, test_df


def filter_congestion_features(
    df: pd.DataFrame,
    prohibited: frozenset[str] = PROHIBITED_CONGESTION_COLUMNS,
) -> pd.DataFrame:
    """Filter out identifiers, target columns, and prohibited timestamp fields."""
    cols_to_drop = [c for c in df.columns if c in prohibited or c.startswith("target_")]
    return df.drop(columns=cols_to_drop, errors="ignore")


class LastValueCongestionBaseline(BaseEstimator, RegressorMixin):
    """Last-Value (Persistence) baseline forecaster.

    Predicts future active census at any horizon to be equal to current_active_census:
        y_hat(T + h) = current_active_census(T)
    """

    def __init__(self, horizons: Sequence[int] = (30, 60, 120)) -> None:
        self.horizons = tuple(horizons)
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> LastValueCongestionBaseline:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame; got {type(X).__name__}")
        if "current_active_census" not in X.columns:
            raise ValueError(
                "X must contain 'current_active_census' feature for LastValueBaseline."
            )
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise NotFittedError("LastValueCongestionBaseline is not fitted.")
        if "current_active_census" not in X.columns:
            raise ValueError("X must contain 'current_active_census'.")

        census = X["current_active_census"].astype(float).values
        predictions = {f"pred_census_{h}m": census for h in self.horizons}
        return pd.DataFrame(predictions, index=X.index)


class TimeOfDayMedianCongestionBaseline(BaseEstimator, RegressorMixin):
    """Historical diurnal baseline forecaster.

    Calculates median historical active census grouped by (day_of_week, hour)
    strictly from training observations. Predicts the historical median for
    the future horizon timestamp's diurnal period.
    """

    def __init__(self, horizons: Sequence[int] = (30, 60, 120)) -> None:
        self.horizons = tuple(horizons)
        self.is_fitted_ = False
        self.diurnal_medians_: dict[tuple[int, int], float] = {}
        self.global_median_: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | None = None,
    ) -> TimeOfDayMedianCongestionBaseline:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame; got {type(X).__name__}")
        if "current_active_census" not in X.columns:
            raise ValueError("X must contain 'current_active_census'.")

        df = X.copy()
        if "snapshot_time" in df.columns:
            times = pd.to_datetime(df["snapshot_time"])
            dows = times.dt.dayofweek
            hours = times.dt.hour
        elif "snapshot_day_of_week" in df.columns and "snapshot_hour" in df.columns:
            dows = df["snapshot_day_of_week"].astype(int)
            hours = df["snapshot_hour"].astype(int)
        else:
            raise ValueError(
                "X must contain 'snapshot_time' or ('snapshot_day_of_week', 'hour')."
            )

        df["_dow"] = dows
        df["_hour"] = hours
        grouped = df.groupby(["_dow", "_hour"])["current_active_census"].median()
        self.diurnal_medians_ = {
            (int(k[0]), int(k[1])): float(v) for k, v in grouped.items()
        }
        self.global_median_ = float(df["current_active_census"].median())
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise NotFittedError("TimeOfDayMedianCongestionBaseline is not fitted.")

        df = X.copy()
        has_snap_time = "snapshot_time" in df.columns
        if has_snap_time:
            snap_times = pd.to_datetime(df["snapshot_time"])

        predictions: dict[str, list[float]] = {
            f"pred_census_{h}m": [] for h in self.horizons
        }

        for idx in range(len(df)):
            row = df.iloc[idx]
            for h in self.horizons:
                if has_snap_time:
                    future_dt = snap_times.iloc[idx] + timedelta(minutes=h)
                    future_dow = int(future_dt.dayofweek)
                    future_hour = int(future_dt.hour)
                else:
                    curr_hour = int(row.get("snapshot_hour", 12))
                    curr_dow = int(row.get("snapshot_day_of_week", 0))
                    added_hours = h // 60
                    future_hour = (curr_hour + added_hours) % 24
                    added_days = (curr_hour + added_hours) // 24
                    future_dow = (curr_dow + added_days) % 7

                pred = self.diurnal_medians_.get(
                    (future_dow, future_hour), self.global_median_
                )
                predictions[f"pred_census_{h}m"].append(pred)

        return pd.DataFrame(predictions, index=X.index)


class RidgeCongestionModel(BaseEstimator, RegressorMixin):
    """Direct multi-horizon L2-regularized linear congestion forecaster."""

    def __init__(
        self,
        horizons: Sequence[int] = (30, 60, 120),
        alpha: float = 1.0,
    ) -> None:
        self.horizons = tuple(horizons)
        self.alpha = alpha
        self.models_: dict[int, Pipeline] = {}
        self.feature_names_in_: list[str] = []
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> RidgeCongestionModel:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame; got {type(X).__name__}")
        if not isinstance(y, pd.DataFrame):
            raise TypeError(f"y must be a pd.DataFrame; got {type(y).__name__}")

        X_clean = filter_congestion_features(X)
        self.feature_names_in_ = list(X_clean.columns)

        for h in self.horizons:
            target_col = f"target_census_{h}m"
            if target_col not in y.columns:
                raise KeyError(f"Target column '{target_col}' not found in y.")

            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=self.alpha, random_state=42)),
                ]
            )
            pipe.fit(X_clean, y[target_col])
            self.models_[h] = pipe

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise NotFittedError("RidgeCongestionModel is not fitted.")

        X_clean = filter_congestion_features(X)[self.feature_names_in_]
        predictions: dict[str, np.ndarray] = {}

        for h in self.horizons:
            preds = self.models_[h].predict(X_clean)
            # Physical non-negative census projection
            predictions[f"pred_census_{h}m"] = np.clip(preds, 0.0, None)

        return pd.DataFrame(predictions, index=X.index)


class XGBoostCongestionModel(BaseEstimator, RegressorMixin):
    """Direct multi-horizon Gradient Boosted Tree congestion forecaster."""

    def __init__(
        self,
        horizons: Sequence[int] = (30, 60, 120),
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.horizons = tuple(horizons)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models_: dict[int, xgb.XGBRegressor] = {}
        self.feature_names_in_: list[str] = []
        self.is_fitted_ = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> XGBoostCongestionModel:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame; got {type(X).__name__}")
        if not isinstance(y, pd.DataFrame):
            raise TypeError(f"y must be a pd.DataFrame; got {type(y).__name__}")

        X_clean = filter_congestion_features(X)
        self.feature_names_in_ = list(X_clean.columns)

        for h in self.horizons:
            target_col = f"target_census_{h}m"
            if target_col not in y.columns:
                raise KeyError(f"Target column '{target_col}' not found in y.")

            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=1,
            )
            model.fit(X_clean, y[target_col])
            self.models_[h] = model

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise NotFittedError("XGBoostCongestionModel is not fitted.")

        X_clean = filter_congestion_features(X)[self.feature_names_in_]
        predictions: dict[str, np.ndarray] = {}

        for h in self.horizons:
            preds = self.models_[h].predict(X_clean)
            # Physical non-negative census projection
            predictions[f"pred_census_{h}m"] = np.clip(preds, 0.0, None)

        return pd.DataFrame(predictions, index=X.index)

    def get_feature_importances(self, horizon: int = 60) -> dict[str, float]:
        """Get Gini feature importances for a specific forecast horizon."""
        if not self.is_fitted_:
            raise NotFittedError("XGBoostCongestionModel is not fitted.")
        if horizon not in self.models_:
            raise ValueError(
                f"Horizon {horizon} not in fitted horizons {self.horizons}"
            )

        importances = self.models_[horizon].feature_importances_
        return {
            name: round(float(imp), 5)
            for name, imp in zip(self.feature_names_in_, importances)
        }


class CongestionPredictor:
    """Production container for Emergency Department congestion forecasting."""

    def __init__(
        self,
        model: Any,
        horizons: Sequence[int] = (30, 60, 120),
        feature_names: list[str] | None = None,
        calibrators: dict[int, ConformalIntervalCalibrator] | None = None,
        model_name: str = "xgboost_congestion",
        model_version: str = "0.1.0",
    ) -> None:
        self.model = model
        self.horizons = tuple(horizons)
        self.feature_names = feature_names or getattr(model, "feature_names_in_", [])
        self.calibrators = calibrators or {}
        self.model_name = model_name
        self.model_version = model_version

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Batch predict multi-horizon future active census."""
        return self.model.predict(X)

    def predict_single(
        self,
        snapshot_features: dict[str, Any] | pd.Series,
        coverage_level: float = 0.90,
    ) -> dict[str, Any]:
        """Generate structured forecast and operational indicators for a snapshot."""
        feats = (
            snapshot_features.to_dict()
            if isinstance(snapshot_features, pd.Series)
            else dict(snapshot_features)
        )
        df_single = pd.DataFrame([feats])

        pred_df = self.model.predict(df_single)
        current_census = int(feats.get("current_active_census", 0))

        forecasts: dict[str, Any] = {}
        for h in self.horizons:
            pred_col = f"pred_census_{h}m"
            pred_val = round(float(pred_df[pred_col].iloc[0]), 1)

            interval_payload = None
            if h in self.calibrators and self.calibrators[h].is_calibrated:
                raw_interval = self.calibrators[h].get_interval_for_prediction(
                    prediction=pred_val,
                    coverage_level=coverage_level,
                )
                interval_payload = {
                    "lower_census": raw_interval["lower_minutes"],
                    "upper_census": raw_interval["upper_minutes"],
                    "lower_minutes": raw_interval["lower_minutes"],
                    "upper_minutes": raw_interval["upper_minutes"],
                    "coverage_level": raw_interval["coverage_level"],
                    "method": raw_interval["method"],
                }

            forecasts[f"{h}m"] = {
                "horizon_minutes": h,
                "predicted_census": pred_val,
                "prediction_interval": interval_payload,
            }

        state_payload = classify_congestion_state(current_census)
        bottleneck_payload = detect_bottleneck_indicators(feats)

        return {
            "current_active_census": current_census,
            "forecasts": forecasts,
            "congestion_state": state_payload,
            "bottleneck_indicators": bottleneck_payload,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

    def save(self, filepath: str | Path) -> None:
        """Serialize CongestionPredictor container to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved CongestionPredictor to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> CongestionPredictor:
        """Deserialize CongestionPredictor container from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Predictor file not found: {path}")
        predictor = joblib.load(path)
        if not isinstance(predictor, cls):
            raise TypeError(
                f"Loaded object is not a CongestionPredictor; got {type(predictor)}"
            )
        return predictor
