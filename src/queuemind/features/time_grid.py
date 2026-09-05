"""Department-level time-series snapshot grid and congestion target generation.

Constructs regular time-series snapshot grids and extracts department-level
operational state features strictly respecting point-in-time boundaries:
- Active patient census
- Rolling arrival and departure velocities
- Net flow rates and ratios
- Acuity mix of active patients
- Multi-horizon future active census targets (e.g. +30m, +60m, +120m)
"""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_time_grid(
    start_time: pd.Timestamp | str,
    end_time: pd.Timestamp | str,
    step_minutes: int = 15,
) -> list[pd.Timestamp]:
    """Generate an ordered, regular sequence of timestamps.

    Args:
        start_time: Initial grid timestamp.
        end_time: Final grid timestamp boundary (inclusive).
        step_minutes: Grid interval step in minutes (must be > 0).

    Returns:
        List of Timestamp objects spaced by step_minutes.

    Raises:
        ValueError: If step_minutes <= 0 or start_time > end_time.
    """
    if step_minutes <= 0:
        raise ValueError(f"step_minutes must be positive; got {step_minutes}")

    t_start = pd.to_datetime(start_time)
    t_end = pd.to_datetime(end_time)

    if t_start > t_end:
        raise ValueError(
            f"start_time ({t_start}) cannot be strictly after end_time ({t_end})"
        )

    grid: list[pd.Timestamp] = []
    curr = t_start
    step_delta = timedelta(minutes=step_minutes)

    while curr <= t_end:
        grid.append(curr)
        curr += step_delta

    return grid


def extract_department_snapshot_features(
    snapshot_time: pd.Timestamp | str,
    edstays_df: pd.DataFrame,
    triage_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Extract department-level operational features strictly at snapshot_time.

    HARD INFORMATION BOUNDARY:
    - Active census: Encounters with intime <= snapshot_time < outtime.
    - Rolling arrivals: Arrivals in (snapshot_time - delta, snapshot_time].
    - Rolling departures: Departures in (snapshot_time - delta, snapshot_time].
    - Future arrivals, future departures, and future events (> snapshot_time) are
      never observed or accessed.

    Args:
        snapshot_time: Hard information cutoff timestamp.
        edstays_df: Master encounters DataFrame containing 'intime' and 'outtime'.
        triage_df: Optional triage DataFrame containing 'stay_id' and 'acuity'.

    Returns:
        Dictionary of snapshot-safe department flow features.
    """
    snap_dt = pd.to_datetime(snapshot_time)

    if "intime" not in edstays_df.columns or "outtime" not in edstays_df.columns:
        logger.warning("edstays_df missing intime or outtime; returning zero features")
        return {
            "current_active_census": 0,
            "recent_arrivals_15m": 0,
            "recent_arrivals_30m": 0,
            "recent_arrivals_60m": 0,
            "recent_arrivals_120m": 0,
            "recent_departures_15m": 0,
            "recent_departures_30m": 0,
            "recent_departures_60m": 0,
            "recent_departures_120m": 0,
            "net_flow_15m": 0,
            "net_flow_30m": 0,
            "net_flow_60m": 0,
            "arrival_rate_per_hour_60m": 0.0,
            "departure_rate_per_hour_60m": 0.0,
            "arrival_departure_ratio_60m": 1.0,
            "high_acuity_active_count": 0,
            "high_acuity_ratio": 0.0,
            "mean_active_acuity": None,
            "snapshot_hour": snap_dt.hour,
            "snapshot_day_of_week": snap_dt.dayofweek,
            "snapshot_is_weekend": int(snap_dt.dayofweek >= 5),
            "snapshot_hour_sin": round(
                float(np.sin(2.0 * np.pi * snap_dt.hour / 24.0)), 4
            ),
            "snapshot_hour_cos": round(
                float(np.cos(2.0 * np.pi * snap_dt.hour / 24.0)), 4
            ),
            "snapshot_month": snap_dt.month,
        }

    intimes = pd.to_datetime(edstays_df["intime"])
    outtimes = pd.to_datetime(edstays_df["outtime"])

    # Active patient census: present in the department at snapshot_time
    active_mask = (intimes <= snap_dt) & (outtimes > snap_dt)
    active_count = int(active_mask.sum())

    # Preceding rolling time windows
    t_15m = snap_dt - timedelta(minutes=15)
    t_30m = snap_dt - timedelta(minutes=30)
    t_60m = snap_dt - timedelta(minutes=60)
    t_120m = snap_dt - timedelta(minutes=120)

    # Rolling arrivals strictly <= snapshot_time
    arr_15m = int(((intimes >= t_15m) & (intimes <= snap_dt)).sum())
    arr_30m = int(((intimes >= t_30m) & (intimes <= snap_dt)).sum())
    arr_60m = int(((intimes >= t_60m) & (intimes <= snap_dt)).sum())
    arr_120m = int(((intimes >= t_120m) & (intimes <= snap_dt)).sum())

    # Rolling departures strictly <= snapshot_time (completed discharges)
    dep_15m = int(((outtimes >= t_15m) & (outtimes <= snap_dt)).sum())
    dep_30m = int(((outtimes >= t_30m) & (outtimes <= snap_dt)).sum())
    dep_60m = int(((outtimes >= t_60m) & (outtimes <= snap_dt)).sum())
    dep_120m = int(((outtimes >= t_120m) & (outtimes <= snap_dt)).sum())

    # Net flow velocities: arrivals minus departures
    net_15m = arr_15m - dep_15m
    net_30m = arr_30m - dep_30m
    net_60m = arr_60m - dep_60m

    # Flow velocity rates and ratios (Laplace smoothing)
    arr_dep_ratio_60m = round(float(arr_60m + 1.0) / float(dep_60m + 1.0), 4)

    # Acuity mix of active patients
    high_acuity_count = 0
    mean_acuity: float | None = None

    if (
        triage_df is not None
        and not triage_df.empty
        and "stay_id" in edstays_df.columns
        and "stay_id" in triage_df.columns
        and "acuity" in triage_df.columns
    ):
        active_stay_ids = set(edstays_df.loc[active_mask, "stay_id"])
        if active_stay_ids:
            active_triage = triage_df[triage_df["stay_id"].isin(active_stay_ids)]
            valid_acuities = pd.to_numeric(
                active_triage["acuity"], errors="coerce"
            ).dropna()
            if not valid_acuities.empty:
                high_acuity_count = int((valid_acuities <= 2.0).sum())
                mean_acuity = round(float(valid_acuities.mean()), 3)

    high_acuity_ratio = (
        round(float(high_acuity_count) / float(max(1, active_count)), 4)
        if active_count > 0
        else 0.0
    )

    return {
        "current_active_census": active_count,
        "recent_arrivals_15m": arr_15m,
        "recent_arrivals_30m": arr_30m,
        "recent_arrivals_60m": arr_60m,
        "recent_arrivals_120m": arr_120m,
        "recent_departures_15m": dep_15m,
        "recent_departures_30m": dep_30m,
        "recent_departures_60m": dep_60m,
        "recent_departures_120m": dep_120m,
        "net_flow_15m": net_15m,
        "net_flow_30m": net_30m,
        "net_flow_60m": net_60m,
        "arrival_rate_per_hour_60m": float(arr_60m),
        "departure_rate_per_hour_60m": float(dep_60m),
        "arrival_departure_ratio_60m": arr_dep_ratio_60m,
        "high_acuity_active_count": high_acuity_count,
        "high_acuity_ratio": high_acuity_ratio,
        "mean_active_acuity": mean_acuity,
        "snapshot_hour": snap_dt.hour,
        "snapshot_day_of_week": snap_dt.dayofweek,
        "snapshot_is_weekend": int(snap_dt.dayofweek >= 5),
        "snapshot_hour_sin": round(float(np.sin(2.0 * np.pi * snap_dt.hour / 24.0)), 4),
        "snapshot_hour_cos": round(float(np.cos(2.0 * np.pi * snap_dt.hour / 24.0)), 4),
        "snapshot_month": snap_dt.month,
    }


def create_congestion_targets(
    snapshot_time: pd.Timestamp | str,
    edstays_df: pd.DataFrame,
    horizons: Sequence[int] = (30, 60, 120),
) -> dict[str, int]:
    """Calculate future active census targets for snapshot_time across horizons.

    TARGET-SIDE SEPARATION:
    Target calculation queries future active census at (snapshot_time + horizon).
    These targets are strictly segregated from model input features.

    Args:
        snapshot_time: Reference timestamp T.
        edstays_df: Master encounters DataFrame.
        horizons: Forecasting horizons in minutes (e.g., 30, 60, 120).

    Returns:
        Dictionary mapping f"target_census_{h}m" to future active patient count.
    """
    snap_dt = pd.to_datetime(snapshot_time)
    intimes = pd.to_datetime(edstays_df["intime"])
    outtimes = pd.to_datetime(edstays_df["outtime"])

    targets: dict[str, int] = {}
    for h in horizons:
        t_future = snap_dt + timedelta(minutes=h)
        future_active = int(((intimes <= t_future) & (outtimes > t_future)).sum())
        targets[f"target_census_{h}m"] = future_active

    return targets


def generate_congestion_dataset(
    edstays_df: pd.DataFrame,
    triage_df: pd.DataFrame | None = None,
    step_minutes: int = 15,
    horizons: Sequence[int] = (30, 60, 120),
    start_time: pd.Timestamp | str | None = None,
    end_time: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct paired feature and multi-horizon target DataFrames.

    Generates regular time-grid snapshots over [start_time, end_time - max(horizons)].
    For each snapshot:
    - Feature row X: calculated strictly from data <= T.
    - Target row Y: active census at T + h for each h in horizons.

    Args:
        edstays_df: Master encounters DataFrame.
        triage_df: Optional triage DataFrame.
        step_minutes: Grid interval step in minutes.
        horizons: Future forecast horizons in minutes.
        start_time: Optional grid start timestamp. Defaults to earliest intime.
        end_time: Optional grid end timestamp. Defaults to latest outtime.

    Returns:
        Tuple of (X, Y) where:
        - X: DataFrame of snapshot features with 'snapshot_time' column.
        - Y: DataFrame of targets aligned by index (target_census_{h}m).

    Raises:
        ValueError: If edstays_df is empty or valid grid range is negative.
    """
    if edstays_df.empty:
        raise ValueError("Cannot generate congestion dataset from empty edstays_df.")

    intimes = pd.to_datetime(edstays_df["intime"])
    outtimes = pd.to_datetime(edstays_df["outtime"])

    grid_start = pd.to_datetime(start_time) if start_time else intimes.min()
    grid_end = pd.to_datetime(end_time) if end_time else outtimes.max()

    max_horizon = max(horizons) if horizons else 0
    effective_end = grid_end - timedelta(minutes=max_horizon)

    if grid_start > effective_end:
        raise ValueError(
            f"Insufficient data span: start ({grid_start}) is after effective end "
            f"({effective_end}) after reserving {max_horizon}m horizon."
        )

    grid = generate_time_grid(
        start_time=grid_start,
        end_time=effective_end,
        step_minutes=step_minutes,
    )

    feature_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for snap in grid:
        feats = extract_department_snapshot_features(
            snapshot_time=snap,
            edstays_df=edstays_df,
            triage_df=triage_df,
        )
        feats["snapshot_time"] = snap
        targets = create_congestion_targets(
            snapshot_time=snap,
            edstays_df=edstays_df,
            horizons=horizons,
        )

        feature_rows.append(feats)
        target_rows.append(targets)

    X = pd.DataFrame(feature_rows)
    Y = pd.DataFrame(target_rows)

    return X, Y
