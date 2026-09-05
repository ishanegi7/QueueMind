"""
Emergency Department congestion and operational flow feature extraction.

Calculates real-time department operational state, patient census, rolling
arrival/discharge velocities, and acuity load strictly respecting the
point-in-time snapshot boundary.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_congestion_features(
    snapshot_time: pd.Timestamp,
    edstays_df: pd.DataFrame,
    triage_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Calculate department-level flow congestion features at snapshot_time.

    POINT-IN-TIME AVAILABILITY GUARANTEE:
    - Active patients: Encounters registered at or before snapshot_time that have
      not yet completed departure (intime <= snapshot_time < outtime).
    - Recent arrivals: Encounters with intime in:
      (snapshot_time - delta, snapshot_time].
    - Recent departures: Encounters with outtime in:
      (snapshot_time - delta, snapshot_time].
    - Under NO circumstances are departures or events occurring after snapshot_time
      included in historical counts or rolling velocities.

    Args:
        snapshot_time: Hard information cutoff timestamp.
        edstays_df: Master encounters DataFrame containing 'intime' and 'outtime'.
        triage_df: Optional triage DataFrame with 'stay_id' and 'acuity'.

    Returns:
        Dict[str, Any]: Dictionary of operational flow features.
    """
    snap_dt = pd.to_datetime(snapshot_time)

    # Ensure required columns are present
    if "intime" not in edstays_df.columns or "outtime" not in edstays_df.columns:
        logger.warning(
            "edstays_df missing intime or outtime; returning zero congestion"
        )
        return {
            "active_patient_count": 0,
            "recent_arrivals_15m": 0,
            "recent_arrivals_30m": 0,
            "recent_arrivals_60m": 0,
            "recent_departures_15m": 0,
            "recent_departures_30m": 0,
            "recent_departures_60m": 0,
            "arrival_departure_ratio_60m": 1.0,
            "net_flow_60m": 0,
            "high_acuity_active_count": 0,
        }

    intimes = pd.to_datetime(edstays_df["intime"])
    outtimes = pd.to_datetime(edstays_df["outtime"])

    # Active patient census: arrived on or before snapshot, departing after snapshot
    active_mask = (intimes <= snap_dt) & (outtimes > snap_dt)
    active_count = int(active_mask.sum())

    # Rolling arrivals in preceding intervals
    t_15m = snap_dt - timedelta(minutes=15)
    t_30m = snap_dt - timedelta(minutes=30)
    t_60m = snap_dt - timedelta(minutes=60)
    t_120m = snap_dt - timedelta(minutes=120)

    arr_15m = int(((intimes >= t_15m) & (intimes <= snap_dt)).sum())
    arr_30m = int(((intimes >= t_30m) & (intimes <= snap_dt)).sum())
    arr_60m = int(((intimes >= t_60m) & (intimes <= snap_dt)).sum())
    arr_120m = int(((intimes >= t_120m) & (intimes <= snap_dt)).sum())

    # Rolling departures: strictly departures completed ON OR BEFORE snapshot
    dep_15m = int(((outtimes >= t_15m) & (outtimes <= snap_dt)).sum())
    dep_30m = int(((outtimes >= t_30m) & (outtimes <= snap_dt)).sum())
    dep_60m = int(((outtimes >= t_60m) & (outtimes <= snap_dt)).sum())
    dep_120m = int(((outtimes >= t_120m) & (outtimes <= snap_dt)).sum())

    # Flow velocities and ratios (with Laplace smoothing to avoid division by zero)
    arr_dep_ratio_60m = round(float(arr_60m + 1.0) / float(dep_60m + 1.0), 4)
    net_flow_60m = arr_60m - dep_60m

    # Acuity burden of active patients
    high_acuity_active = 0
    mean_active_acuity = None

    if (
        triage_df is not None
        and not triage_df.empty
        and "stay_id" in edstays_df.columns
    ):
        active_stay_ids = set(edstays_df.loc[active_mask, "stay_id"])
        if (
            active_stay_ids
            and "stay_id" in triage_df.columns
            and "acuity" in triage_df.columns
        ):
            active_triage = triage_df[triage_df["stay_id"].isin(active_stay_ids)]
            valid_acuities = pd.to_numeric(
                active_triage["acuity"], errors="coerce"
            ).dropna()
            if not valid_acuities.empty:
                high_acuity_active = int((valid_acuities <= 2.0).sum())
                mean_active_acuity = round(float(valid_acuities.mean()), 3)

    return {
        "active_patient_count": active_count,
        "recent_arrivals_15m": arr_15m,
        "recent_arrivals_30m": arr_30m,
        "recent_arrivals_60m": arr_60m,
        "recent_arrivals_120m": arr_120m,
        "recent_departures_15m": dep_15m,
        "recent_departures_30m": dep_30m,
        "recent_departures_60m": dep_60m,
        "recent_departures_120m": dep_120m,
        "arrival_departure_ratio_60m": arr_dep_ratio_60m,
        "net_flow_60m": net_flow_60m,
        "high_acuity_active_count": high_acuity_active,
        "mean_active_acuity": mean_active_acuity,
    }
