"""Model training and chronological data partitioning module for QueueMind.

This module enforces:
1. Strict chronological train/val/test partitioning to eliminate future leakage.
2. Model input contract enforcement (prohibiting identifiers, timestamps, targets).
3. Preprocessor pipeline construction fitted strictly on training observations.
4. XGBoost regression candidate model training and parameterization.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

PROHIBITED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "stay_id",
        "snapshot_time",
        "intime",
        "outtime",
        "disposition",
        "remaining_time_minutes",
        "remaining_time",
    }
)


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "snapshot_time",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation, and test subsets chronologically.

    Enforces strict chronological partitioning:
        max(train[time_col]) <= min(val[time_col]) <= max(val) <= min(test)

    Args:
        df: Input DataFrame containing snapshots.
        time_col: Timestamp column name to sort by.
        train_ratio: Proportion of observations for training.
        val_ratio: Proportion of observations for validation.
        test_ratio: Proportion of observations for test.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios do not sum to 1.0, are non-positive, or if df is empty.
        KeyError: If time_col is missing from df.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    if time_col not in df.columns:
        raise KeyError(f"Timestamp column '{time_col}' not found in DataFrame.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(
            f"Split ratios must sum to 1.0; got {train_ratio} + {val_ratio} "
            f"+ {test_ratio} = {total_ratio}"
        )

    if min(train_ratio, val_ratio, test_ratio) <= 0.0:
        raise ValueError("Split ratios must be strictly positive.")

    n_samples = len(df)
    n_train = int(np.floor(n_samples * train_ratio))
    n_val = int(np.floor(n_samples * val_ratio))
    n_test = n_samples - (n_train + n_val)

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"Dataset with {n_samples} rows is too small for split ratios "
            f"({train_ratio}, {val_ratio}, {test_ratio}). "
            "All partitions must have >= 1 row."
        )

    # Sort strictly by timestamp ascending
    sorted_df = df.sort_values(by=time_col, ascending=True).reset_index(drop=True)

    train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = sorted_df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # Verify chronological ordering across splits
    t_train_max = pd.to_datetime(train_df[time_col]).max()
    t_val_min = pd.to_datetime(val_df[time_col]).min()
    t_val_max = pd.to_datetime(val_df[time_col]).max()
    t_test_min = pd.to_datetime(test_df[time_col]).min()

    if t_train_max > t_val_min:
        raise ValueError(
            f"Chronological ordering violation: train max ({t_train_max}) "
            f"> val min ({t_val_min})"
        )

    if t_val_max > t_test_min:
        raise ValueError(
            f"Chronological ordering violation: val max ({t_val_max}) "
            f"> test min ({t_test_min})"
        )

    return train_df, val_df, test_df


def get_model_feature_names(
    df: pd.DataFrame,
    target_col: str = "remaining_time_minutes",
    prohibited_cols: set[str] | frozenset[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Extract valid numeric and categorical feature names obeying the input contract.

    Args:
        df: Input DataFrame containing features and targets.
        target_col: Target column name to exclude.
        prohibited_cols: Additional prohibited column names.

    Returns:
        Tuple of (numeric_cols, categorical_cols).
    """
    excluded = set(PROHIBITED_FEATURE_COLUMNS)
    excluded.add(target_col)
    if prohibited_cols:
        excluded.update(prohibited_cols)

    candidate_cols = [c for c in df.columns if c not in excluded]

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Construct an sklearn ColumnTransformer for tabular features.

    Note:
        Numeric columns can either be scaled (for linear models) or passed
        through (for gradient boosted trees that handle raw scales and
        missing values natively).

    Args:
        categorical_cols: List of categorical feature column names.
        numeric_cols: List of numeric feature column names.
        scale_numeric: Whether to apply StandardScaler and median
            imputation to numerics.

    Returns:
        Configured ColumnTransformer instance.
    """
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        if scale_numeric:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, numeric_cols))
        else:
            # XGBoost handles missing values natively; passthrough numeric values
            transformers.append(("num", "passthrough", numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


class XGBoostCandidate(BaseEstimator, RegressorMixin):
    """XGBoost regression candidate model for remaining patient ED duration."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 1,
        clip_predictions_at_zero: bool = True,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clip_predictions_at_zero = clip_predictions_at_zero
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

        self.preprocessor_: ColumnTransformer | None = None
        self.regressor_: xgb.XGBRegressor | None = None
        self.feature_names_in_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray | Sequence[float],
        eval_set: list[tuple[pd.DataFrame, Any]] | None = None,
        verbose: bool = False,
    ) -> XGBoostCandidate:
        """Fit the preprocessor and XGBoost model strictly on training observations."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("XGBoostCandidate requires a pandas DataFrame for X.")

        # Guard against target / prohibited leakage into feature matrix
        leaked_cols = set(X.columns).intersection(PROHIBITED_FEATURE_COLUMNS)
        if leaked_cols:
            raise ValueError(
                "Data leakage detected! Feature matrix contains "
                f"prohibited columns: {leaked_cols}"
            )

        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Cannot fit XGBoostCandidate on empty target array.")

        self.feature_names_in_ = list(X.columns)

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

        self.preprocessor_ = build_preprocessor(
            categorical_cols=cat_cols,
            numeric_cols=num_cols,
            scale_numeric=False,
        )

        # Fit preprocessor strictly on training X
        X_trans = self.preprocessor_.fit_transform(X)

        transformed_eval_set = None
        if eval_set is not None:
            transformed_eval_set = []
            for eval_x, eval_y in eval_set:
                eval_x_trans = self.preprocessor_.transform(eval_x)
                transformed_eval_set.append(
                    (eval_x_trans, np.asarray(eval_y, dtype=float))
                )

        self.regressor_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="reg:squarederror",
        )

        self.regressor_.fit(
            X_trans,
            y_arr,
            eval_set=transformed_eval_set,
            verbose=verbose,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict remaining duration in minutes."""
        if self.regressor_ is None or self.preprocessor_ is None:
            raise NotFittedError("XGBoostCandidate is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("XGBoostCandidate requires a pandas DataFrame for X.")

        # Guard against prohibited leakage
        leaked_cols = set(X.columns).intersection(PROHIBITED_FEATURE_COLUMNS)
        if leaked_cols:
            raise ValueError(
                "Data leakage detected! Feature matrix contains "
                f"prohibited columns: {leaked_cols}"
            )

        X_trans = self.preprocessor_.transform(X)
        preds = self.regressor_.predict(X_trans)

        if self.clip_predictions_at_zero:
            preds = np.clip(preds, a_min=0.0, a_max=None)

        return np.asarray(preds, dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return regressor feature importances."""
        if self.regressor_ is None:
            raise NotFittedError("XGBoostCandidate is not fitted yet.")
        return self.regressor_.feature_importances_
