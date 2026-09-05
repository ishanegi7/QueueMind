"""Prediction uncertainty and split-conformal calibration module for QueueMind.

This module provides split-conformal prediction intervals for patient-flow regression:
1. ConformalIntervalCalibrator: Fits on chronological validation / calibration data
   and derives finite-sample calibrated residual cutoffs.
2. Generates calibrated prediction intervals: [max(0.0, y_hat - q), y_hat + q]
   guaranteeing marginal empirical coverage under exchangeability.
3. Completely isolates the holdout test set from calibration.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


class ConformalIntervalCalibrator:
    """Split-conformal prediction interval calibrator for patient journey duration.

    Statistical Interpretation & Assumptions:
    - Provides a marginal coverage guarantee under the exchangeability assumption:
      P(Y_test in [lower, upper]) >= 1 - alpha.
    - Does NOT claim a subjective 90% individual probability for a single patient.
    - Assumes the calibration set is temporally prior to test set and representative
      of operational conditions.
    """

    def __init__(self, default_coverage_level: float = 0.90) -> None:
        if not (0.0 < default_coverage_level < 1.0):
            raise ValueError(
                f"Default coverage must be in (0, 1); got {default_coverage_level}"
            )
        self.default_coverage_level = default_coverage_level
        self.residuals_: np.ndarray | None = None
        self.n_calibration_samples_: int = 0

    @property
    def is_calibrated(self) -> bool:
        """Return True if calibrator has been fitted on calibration observations."""
        return self.residuals_ is not None and self.n_calibration_samples_ > 0

    def calibrate(
        self,
        y_true: pd.Series | np.ndarray | Sequence[float],
        y_pred: pd.Series | np.ndarray | Sequence[float],
    ) -> ConformalIntervalCalibrator:
        """Calibrate non-conformity scores (absolute residuals) on validation set.

        Args:
            y_true: True duration in minutes from calibration set.
            y_pred: Model predictions in minutes for calibration set.

        Returns:
            Fitted ConformalIntervalCalibrator instance.

        Raises:
            ValueError: If inputs are empty, have mismatched lengths, or contain NaNs.
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        if len(y_t) == 0 or len(y_p) == 0:
            raise ValueError("Cannot calibrate on empty target or prediction arrays.")

        if len(y_t) != len(y_p):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_t)} samples, "
                f"y_pred has {len(y_p)} samples."
            )

        if np.isnan(y_t).any() or np.isnan(y_p).any():
            raise ValueError("Calibration inputs must not contain NaN values.")

        # Absolute residual non-conformity scores
        abs_residuals = np.abs(y_t - y_p)
        self.residuals_ = np.sort(abs_residuals)
        self.n_calibration_samples_ = len(self.residuals_)

        return self

    def get_cutoff(self, coverage_level: float | None = None) -> float:
        """Calculate the calibrated residual threshold for a desired coverage level.

        Uses the standard finite-sample conformal quantile formula:
            k = ceil((n + 1) * coverage_level) - 1
            cutoff = sorted_residuals[k]

        Args:
            coverage_level: Desired coverage probability in (0, 1).

        Returns:
            Residual cutoff q in minutes.

        Raises:
            NotFittedError: If calibrator is not yet calibrated.
            ValueError: If coverage_level is not strictly in (0, 1).
        """
        if not self.is_calibrated or self.residuals_ is None:
            raise NotFittedError(
                "ConformalIntervalCalibrator is not calibrated. Call calibrate() first."
            )

        cov = (
            coverage_level
            if coverage_level is not None
            else self.default_coverage_level
        )
        if not (0.0 < cov < 1.0):
            raise ValueError(f"Coverage level must be in (0, 1); got {cov}")

        n = self.n_calibration_samples_
        # Finite-sample conformal index
        k = int(np.ceil((n + 1) * cov)) - 1
        k = min(max(0, k), n - 1)

        return float(self.residuals_[k])

    def predict_interval(
        self,
        y_pred: float | pd.Series | np.ndarray | Sequence[float],
        coverage_level: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute lower and upper prediction interval bounds for model predictions.

        Enforces physical non-negativity: lower bound is clipped at 0.0 minutes.

        Args:
            y_pred: Predicted remaining minutes (scalar or array).
            coverage_level: Desired coverage level in (0, 1).

        Returns:
            Tuple of (lower_bounds, upper_bounds) as 1D numpy arrays.
        """
        cutoff = self.get_cutoff(coverage_level)
        preds = np.asarray(y_pred, dtype=float)

        lower = np.maximum(0.0, preds - cutoff)
        upper = preds + cutoff

        return lower, upper

    def get_interval_for_prediction(
        self,
        prediction: float,
        coverage_level: float | None = None,
    ) -> dict[str, Any]:
        """Generate structured prediction interval dictionary for a single prediction.

        Args:
            prediction: Scalar predicted minutes.
            coverage_level: Desired coverage level.

        Returns:
            Dictionary with lower_minutes, upper_minutes, coverage_level, and method.
        """
        cov = (
            coverage_level
            if coverage_level is not None
            else self.default_coverage_level
        )
        lower, upper = self.predict_interval(prediction, coverage_level=cov)

        lower_val = (
            float(lower[0])
            if hasattr(lower, "__len__") and lower.ndim > 0
            else float(lower)
        )
        upper_val = (
            float(upper[0])
            if hasattr(upper, "__len__") and upper.ndim > 0
            else float(upper)
        )

        return {
            "lower_minutes": round(lower_val, 2),
            "upper_minutes": round(upper_val, 2),
            "coverage_level": float(cov),
            "method": "split_conformal",
        }
