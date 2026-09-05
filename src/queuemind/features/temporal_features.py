"""
Temporal feature extraction module for QueueMind.

Extracts calendar, diurnal, cyclical, and interval features representing
the operational temporal state at snapshot time. All timestamp handling is
timezone-naive to ensure consistency with MIMIC-IV-ED de-identified dates.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_shift_bucket(hour: int) -> str:
    """
    Classify an hour (0-23) into standard emergency department nursing shifts.

    Shifts:
    - Day/Morning: 07:00 - 14:59 (high arrival volume, full staffing)
    - Evening: 15:00 - 22:59 (peak arrival volume, transition to night)
    - Night: 23:00 - 06:59 (lower arrival volume, reduced staff capacity)

    Args:
        hour: Hour of the day (0 to 23).

    Returns:
        str: Shift name ('morning', 'evening', or 'night').
    """
    if 7 <= hour < 15:
        return "morning"
    elif 15 <= hour < 23:
        return "evening"
    return "night"


def extract_temporal_features(
    snapshot_time: pd.Timestamp,
    arrival_time: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    Extract point-in-time temporal features for a prediction snapshot.

    Args:
        snapshot_time: Hard information cutoff timestamp.
        arrival_time: Optional presentation timestamp (intime).

    Returns:
        Dict[str, Any]: Dictionary of temporal covariates.

    Raises:
        ValueError: If arrival_time is provided and strictly follows snapshot_time.
    """
    snap_dt = pd.to_datetime(snapshot_time)

    hour = int(snap_dt.hour)
    day_of_week = int(snap_dt.dayofweek)  # 0=Monday, 6=Sunday
    is_weekend = int(day_of_week >= 5)
    month = int(snap_dt.month)

    # Cyclical hour encoding: preserves smooth 23:59 -> 00:00 continuity
    hour_radians = 2.0 * np.pi * hour / 24.0
    hour_sin = float(np.sin(hour_radians))
    hour_cos = float(np.cos(hour_radians))

    features: Dict[str, Any] = {
        "snapshot_hour": hour,
        "snapshot_day_of_week": day_of_week,
        "snapshot_is_weekend": is_weekend,
        "snapshot_month": month,
        "snapshot_hour_sin": round(hour_sin, 6),
        "snapshot_hour_cos": round(hour_cos, 6),
        "snapshot_shift": get_shift_bucket(hour),
    }

    if arrival_time is not None:
        arr_dt = pd.to_datetime(arrival_time)
        if snap_dt < arr_dt:
            msg = f"Snapshot time ({snap_dt}) precedes arrival time ({arr_dt})."
            logger.error(msg)
            raise ValueError(msg)

        arr_hour = int(arr_dt.hour)
        arr_dow = int(arr_dt.dayofweek)
        elapsed_minutes = (snap_dt - arr_dt).total_seconds() / 60.0

        features["arrival_hour"] = arr_hour
        features["arrival_day_of_week"] = arr_dow
        features["arrival_is_weekend"] = int(arr_dow >= 5)
        features["arrival_shift"] = get_shift_bucket(arr_hour)
        features["elapsed_time_minutes"] = elapsed_minutes

    return features
