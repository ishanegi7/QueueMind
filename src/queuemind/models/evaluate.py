"""Evaluation metrics and subgroup analysis module for QueueMind patient-flow models.

This module provides standardized regression evaluation functions:
1. Core regression metrics: MAE (minutes), RMSE (minutes), MedAE (minutes),
   and R-squared.
2. Subgroup breakdown evaluation: Calculates metrics partitioned by operational
   cohorts (e.g., triage acuity, nursing shift, arrival transport mode).
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def evaluate_regression(
    y_true: pd.Series | np.ndarray | Sequence[float],
    y_pred: pd.Series | np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Calculate regression performance metrics for patient journey duration.

    Args:
        y_true: True remaining duration in minutes.
        y_pred: Predicted remaining duration in minutes.

    Returns:
        Dictionary with keys 'mae', 'rmse', 'medae', 'r2' with unrounded float values.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Cannot evaluate metrics on empty arrays.")

    if len(y_t) != len(y_p):
        raise ValueError(
            f"Input length mismatch: y_true has {len(y_t)} items, "
            f"y_pred has {len(y_p)} items."
        )

    # Filter any potential NaNs in evaluation arrays
    valid_mask = ~(np.isnan(y_t) | np.isnan(y_p))
    if not np.any(valid_mask):
        raise ValueError("All pairs contain NaN values.")

    y_t_valid = y_t[valid_mask]
    y_p_valid = y_p[valid_mask]

    mae = float(mean_absolute_error(y_t_valid, y_p_valid))
    mse = float(mean_squared_error(y_t_valid, y_p_valid))
    rmse = float(np.sqrt(mse))
    medae = float(median_absolute_error(y_t_valid, y_p_valid))

    if len(y_t_valid) > 1 and np.var(y_t_valid) > 0:
        r2 = float(r2_score(y_t_valid, y_p_valid))
    else:
        r2 = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "r2": r2,
    }


def evaluate_subgroups(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    subgroup_col: str,
) -> pd.DataFrame:
    """Evaluate regression metrics across subgroups/cohorts.

    Args:
        df: DataFrame containing actuals, predictions, and subgroup column.
        y_true_col: Column name of actual remaining time.
        y_pred_col: Column name of predicted remaining time.
        subgroup_col: Column name defining the subgroup/stratum.

    Returns:
        DataFrame summarizing metrics per subgroup level.

    Raises:
        KeyError: If any of the required columns are missing.
        ValueError: If df is empty.
    """
    if df.empty:
        raise ValueError("Cannot evaluate subgroups on an empty DataFrame.")

    for col in (y_true_col, y_pred_col, subgroup_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    rows: list[dict[str, Any]] = []

    # Include NaN groups explicitly
    for group_val, group_df in df.groupby(subgroup_col, dropna=False):
        group_label = "Missing" if pd.isna(group_val) else str(group_val)
        y_t = group_df[y_true_col].to_numpy()
        y_p = group_df[y_pred_col].to_numpy()

        if len(group_df) == 0:
            continue

        metrics = evaluate_regression(y_t, y_p)
        rows.append(
            {
                subgroup_col: group_label,
                "sample_count": len(group_df),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "medae": metrics["medae"],
                "r2": metrics["r2"],
            }
        )

    return pd.DataFrame(rows)


def format_metrics_summary(metrics: dict[str, float], prefix: str = "") -> str:
    """Format regression metrics for human-readable logging.

    Args:
        metrics: Dictionary containing 'mae', 'rmse', 'medae', 'r2'.
        prefix: Optional prefix for lines.

    Returns:
        Multi-line string representation.
    """
    p = f"{prefix} " if prefix else ""
    return (
        f"{p}MAE:   {metrics['mae']:.2f} min\n"
        f"{p}RMSE:  {metrics['rmse']:.2f} min\n"
        f"{p}MedAE: {metrics['medae']:.2f} min\n"
        f"{p}R²:    {metrics['r2']:.4f}"
    )


def evaluate_prediction_intervals(
    y_true: pd.Series | np.ndarray | Sequence[float],
    lower_bounds: pd.Series | np.ndarray | Sequence[float],
    upper_bounds: pd.Series | np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Evaluate prediction interval empirical coverage and widths.

    Args:
        y_true: Ground-truth target values.
        lower_bounds: Predicted lower interval bounds.
        upper_bounds: Predicted upper interval bounds.

    Returns:
        Dictionary with 'empirical_coverage', 'mean_width', and 'median_width'.

    Raises:
        ValueError: If inputs have mismatched lengths or are empty.
    """
    y_t = np.asarray(y_true, dtype=float)
    low = np.asarray(lower_bounds, dtype=float)
    high = np.asarray(upper_bounds, dtype=float)

    if len(y_t) == 0 or len(low) == 0 or len(high) == 0:
        raise ValueError("Cannot evaluate intervals on empty arrays.")

    if not (len(y_t) == len(low) == len(high)):
        raise ValueError(
            f"Input length mismatch: y_true ({len(y_t)}), lower ({len(low)}), "
            f"upper ({len(high)})."
        )

    valid_mask = ~(np.isnan(y_t) | np.isnan(low) | np.isnan(high))
    if not np.any(valid_mask):
        raise ValueError("All interval evaluation pairs contain NaN values.")

    y_valid = y_t[valid_mask]
    low_valid = low[valid_mask]
    high_valid = high[valid_mask]

    covered = (y_valid >= low_valid) & (y_valid <= high_valid)
    empirical_coverage = float(np.mean(covered))

    widths = high_valid - low_valid
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    return {
        "empirical_coverage": empirical_coverage,
        "mean_width": mean_width,
        "median_width": median_width,
    }


def evaluate_interval_subgroups(
    df: pd.DataFrame,
    y_true_col: str,
    lower_col: str,
    upper_col: str,
    subgroup_col: str,
) -> pd.DataFrame:
    """Evaluate prediction interval metrics across operational subgroups.

    Args:
        df: DataFrame containing targets, interval bounds, and subgroup column.
        y_true_col: Target column name.
        lower_col: Lower bound column name.
        upper_col: Upper bound column name.
        subgroup_col: Subgroup category column name.

    Returns:
        DataFrame summarizing coverage and widths per subgroup.

    Raises:
        KeyError: If any required column is missing.
        ValueError: If DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot evaluate interval subgroups on empty DataFrame.")

    for col in (y_true_col, lower_col, upper_col, subgroup_col):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    rows: list[dict[str, Any]] = []
    for group_val, group_df in df.groupby(subgroup_col, dropna=False):
        group_label = "Missing" if pd.isna(group_val) else str(group_val)
        if len(group_df) == 0:
            continue

        metrics = evaluate_prediction_intervals(
            group_df[y_true_col].to_numpy(),
            group_df[lower_col].to_numpy(),
            group_df[upper_col].to_numpy(),
        )
        rows.append(
            {
                subgroup_col: group_label,
                "sample_count": len(group_df),
                "empirical_coverage": metrics["empirical_coverage"],
                "mean_width": metrics["mean_width"],
                "median_width": metrics["median_width"],
            }
        )

    return pd.DataFrame(rows)


def evaluate_congestion_forecasts(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    horizons: Sequence[int] = (30, 60, 120),
) -> dict[str, dict[str, float]]:
    """Evaluate multi-horizon active census forecasts against actual ground truth.

    Computes MAE (patients), RMSE (patients), MedAE (patients), and R² separately
    for each horizon (e.g. 30m, 60m, 120m).

    Args:
        y_true: DataFrame containing 'target_census_{h}m' columns.
        y_pred: DataFrame containing 'pred_census_{h}m' columns.
        horizons: Sequence of forecast horizons in minutes.

    Returns:
        Dictionary mapping horizon string ('30m', '60m', etc.) to regression metrics.

    Raises:
        ValueError: If DataFrames have mismatched lengths or are empty.
        KeyError: If required horizon columns are missing.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} rows, y_pred has {len(y_pred)}."
        )
    if y_true.empty:
        raise ValueError("Cannot evaluate congestion forecasts on empty DataFrames.")

    results: dict[str, dict[str, float]] = {}
    for h in horizons:
        target_col = f"target_census_{h}m"
        pred_col = f"pred_census_{h}m"

        if target_col not in y_true.columns:
            raise KeyError(f"Target column '{target_col}' not found in y_true.")
        if pred_col not in y_pred.columns:
            raise KeyError(f"Prediction column '{pred_col}' not found in y_pred.")

        metrics = evaluate_regression(
            y_true[target_col].to_numpy(),
            y_pred[pred_col].to_numpy(),
        )
        results[f"{h}m"] = metrics

    return results


def format_congestion_metrics_summary(
    results_by_model: dict[str, dict[str, dict[str, float]]],
    horizons: Sequence[int] = (30, 60, 120),
) -> str:
    """Format multi-model multi-horizon congestion evaluation into a markdown table.

    Args:
        results_by_model: Dict mapping model_name -> horizon_str -> metrics_dict.
        horizons: Horizons to include in summary.

    Returns:
        Formatted markdown table string.
    """
    lines = [
        "| Horizon | Model | MAE (pts) | RMSE (pts) | MedAE (pts) | R² |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ]

    for h in horizons:
        h_str = f"{h}m"
        for model_name, model_results in results_by_model.items():
            if h_str in model_results:
                m = model_results[h_str]
                r2_str = f"{m['r2']:.3f}" if not np.isnan(m["r2"]) else "N/A"
                lines.append(
                    f"| {h_str} | {model_name} | {m['mae']:.2f} | "
                    f"{m['rmse']:.2f} | {m['medae']:.2f} | {r2_str} |"
                )

    return "\n".join(lines)
