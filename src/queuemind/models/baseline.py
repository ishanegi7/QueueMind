"""Baseline forecasting models for QueueMind patient-flow regression.

This module provides operational baseline benchmarks for predicting
`remaining_time_minutes` at point-in-time snapshots:
1. GlobalMedianBaseline: Predicts overall training cohort median duration.
2. AcuityStratifiedMedianBaseline: Predicts triage-acuity-stratified median duration,
   with global median fallback for unseen, missing, or out-of-range acuity levels.
3. RidgeRegressionBaseline: Regularized linear regression pipeline with standard
   scaling and one-hot encoding.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class GlobalMedianBaseline(BaseEstimator, RegressorMixin):
    """Predicts the global median remaining time observed in the training cohort."""

    def __init__(self) -> None:
        self.median_remaining_time_: float | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | Sequence[float],
    ) -> GlobalMedianBaseline:
        """Fit global median remaining time from training labels."""
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Cannot fit GlobalMedianBaseline on empty target array.")

        self.median_remaining_time_ = float(np.median(y_arr[~np.isnan(y_arr)]))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict global median remaining time for all samples."""
        if self.median_remaining_time_ is None:
            raise NotFittedError("GlobalMedianBaseline is not fitted yet.")

        n_samples = len(X)
        return np.full(
            shape=(n_samples,), fill_value=self.median_remaining_time_, dtype=float
        )


class AcuityStratifiedMedianBaseline(BaseEstimator, RegressorMixin):
    """Predicts remaining time based on triage acuity group median."""

    def __init__(self, acuity_col: str = "acuity") -> None:
        self.acuity_col = acuity_col
        self.acuity_medians_: dict[int | float, float] = {}
        self.global_median_: float | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | Sequence[float],
    ) -> AcuityStratifiedMedianBaseline:
        """Fit stratified median per triage acuity level."""
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError(
                "Cannot fit AcuityStratifiedMedianBaseline on empty target array."
            )

        valid_y = y_arr[~np.isnan(y_arr)]
        if len(valid_y) == 0:
            raise ValueError("Target array contains only NaN values.")

        self.global_median_ = float(np.median(valid_y))
        self.acuity_medians_ = {}

        if isinstance(X, pd.DataFrame):
            if self.acuity_col not in X.columns:
                raise KeyError(f"Acuity column '{self.acuity_col}' not found in X.")
            acuity_series = X[self.acuity_col]
        elif isinstance(X, np.ndarray):
            # Assume 1D array or first column is acuity
            acuity_series = pd.Series(X if X.ndim == 1 else X[:, 0])
        else:
            acuity_series = pd.Series(X)

        df_fit = pd.DataFrame({"acuity": acuity_series, "y": y_arr}).dropna()

        for acuity_val, group in df_fit.groupby("acuity"):
            if len(group) > 0:
                self.acuity_medians_[acuity_val] = float(group["y"].median())

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict acuity-stratified median remaining time."""
        if self.global_median_ is None:
            raise NotFittedError("AcuityStratifiedMedianBaseline is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            if self.acuity_col not in X.columns:
                raise KeyError(f"Acuity column '{self.acuity_col}' not found in X.")
            acuity_series = X[self.acuity_col]
        elif isinstance(X, np.ndarray):
            acuity_series = pd.Series(X if X.ndim == 1 else X[:, 0])
        else:
            acuity_series = pd.Series(X)

        preds = np.empty(len(acuity_series), dtype=float)
        for i, val in enumerate(acuity_series):
            if pd.isna(val):
                preds[i] = self.global_median_
            elif val in self.acuity_medians_:
                preds[i] = self.acuity_medians_[val]
            else:
                # Unseen acuity in inference falls back safely to training global median
                preds[i] = self.global_median_

        return preds


class RidgeRegressionBaseline(BaseEstimator, RegressorMixin):
    """Linear regression baseline with L2 regularization and standard preprocessing."""

    def __init__(
        self,
        alpha: float = 1.0,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.pipeline_: Pipeline | None = None
        self.feature_names_in_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | Sequence[float],
    ) -> RidgeRegressionBaseline:
        """Fit preprocessor and Ridge model on training set."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RidgeRegressionBaseline requires a pandas DataFrame for X."
            )

        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError(
                "Cannot fit RidgeRegressionBaseline on empty target array."
            )

        self.feature_names_in_ = list(X.columns)

        # Infer columns if not explicitly provided
        num_cols = (
            self.numeric_cols
            if self.numeric_cols is not None
            else list(X.select_dtypes(include=[np.number, "boolean"]).columns)
        )
        cat_cols = (
            self.categorical_cols
            if self.categorical_cols is not None
            else list(X.select_dtypes(include=["object", "category", "string"]).columns)
        )

        transformers: list[tuple[str, Any, list[str]]] = []
        if num_cols:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, num_cols))

        if cat_cols:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ohe",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipe, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline_ = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=self.alpha, random_state=42)),
            ]
        )

        self.pipeline_.fit(X, y_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict remaining minutes using fitted Ridge pipeline."""
        if self.pipeline_ is None:
            raise NotFittedError("RidgeRegressionBaseline is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "RidgeRegressionBaseline requires a pandas DataFrame for X."
            )

        preds = self.pipeline_.predict(X)
        return np.asarray(preds, dtype=float)
