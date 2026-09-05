"""
Point-in-time patient snapshot and target generation module.

Implements the hard information boundary mechanism ensuring that all features
at snapshot time T strictly use information known at or before T.
Target variables (remaining_time_minutes, outtime) are kept strictly segregated.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from queuemind.features.congestion_features import calculate_congestion_features
from queuemind.features.patient_features import extract_patient_features
from queuemind.features.temporal_features import extract_temporal_features

logger = logging.getLogger(__name__)


def create_remaining_time_label(
    outtime: pd.Timestamp,
    snapshot_time: pd.Timestamp,
) -> float:
    """
    Calculate the target variable: remaining ED journey duration in minutes.

    LEAKAGE WARNING:
    This function computes the TARGET label for training and evaluation.
    Its output must NEVER be provided as an input feature to models.

    Args:
        outtime: Actual recorded patient departure timestamp.
        snapshot_time: Point-in-time prediction timestamp.

    Returns:
        float: Remaining stay duration in minutes (>= 0.0).

    Raises:
        ValueError: If snapshot_time is strictly after outtime.
    """
    out_dt = pd.to_datetime(outtime)
    snap_dt = pd.to_datetime(snapshot_time)

    if snap_dt > out_dt:
        msg = (
            f"Snapshot time ({snap_dt}) is after patient departure outtime ({out_dt})."
        )
        logger.error(msg)
        raise ValueError(msg)

    return float((out_dt - snap_dt).total_seconds() / 60.0)


def create_patient_snapshot(
    stay_id: int,
    snapshot_time: pd.Timestamp,
    edstays_df: pd.DataFrame,
    triage_df: Optional[pd.DataFrame] = None,
    vitalsign_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Generate a complete point-in-time feature dictionary for a patient.

    HARD INFORMATION BOUNDARY:
    - Combines patient, temporal, and department congestion features.
    - All queries enforce t <= snapshot_time.
    - Target variables (outtime, remaining_time_minutes, disposition) are
      guaranteed to be ABSENT from the returned dictionary.

    Args:
        stay_id: Encounter identifier.
        snapshot_time: Hard information cutoff timestamp.
        edstays_df: Cleaned edstays DataFrame.
        triage_df: Optional cleaned triage DataFrame.
        vitalsign_df: Optional cleaned vitalsign DataFrame.

    Returns:
        Dict[str, Any]: Prediction feature dictionary at snapshot_time.

    Raises:
        ValueError: If stay_id is not found or snapshot_time < intime.
    """
    snap_dt = pd.to_datetime(snapshot_time)

    # Locate patient encounter
    patient_records = edstays_df[edstays_df["stay_id"] == stay_id]
    if patient_records.empty:
        raise ValueError(f"stay_id {stay_id} not found in edstays_df.")

    patient_row = patient_records.iloc[0]
    intime = pd.to_datetime(patient_row["intime"])

    if snap_dt < intime:
        msg = (
            f"Snapshot time ({snap_dt}) precedes intime ({intime}) "
            f"for stay_id {stay_id}."
        )
        raise ValueError(msg)

    # Extract triage info if available
    triage_row = None
    if triage_df is not None and not triage_df.empty and "stay_id" in triage_df.columns:
        t_records = triage_df[triage_df["stay_id"] == stay_id]
        if not t_records.empty:
            triage_row = t_records.iloc[0]

    # Filter patient vitals history strictly <= snapshot_time
    patient_vitals = None
    if (
        vitalsign_df is not None
        and not vitalsign_df.empty
        and "stay_id" in vitalsign_df.columns
    ):
        pv = vitalsign_df[vitalsign_df["stay_id"] == stay_id]
        if not pv.empty and "charttime" in pv.columns:
            patient_vitals = pv[pd.to_datetime(pv["charttime"]) <= snap_dt]

    # 1. Patient features
    patient_feats = extract_patient_features(
        stay_id=stay_id,
        snapshot_time=snap_dt,
        intime=intime,
        triage_row=triage_row,
        vitals_history=patient_vitals,
        arrival_transport=patient_row.get("arrival_transport"),
        gender=patient_row.get("gender"),
    )

    # 2. Temporal features
    temporal_feats = extract_temporal_features(
        snapshot_time=snap_dt,
        arrival_time=intime,
    )

    # 3. Department congestion features
    congestion_feats = calculate_congestion_features(
        snapshot_time=snap_dt,
        edstays_df=edstays_df,
        triage_df=triage_df,
    )

    # Combine all feature sets
    snapshot: Dict[str, Any] = {
        "stay_id": stay_id,
        "snapshot_time": snap_dt,
    }
    snapshot.update(patient_feats)
    snapshot.update(temporal_feats)
    snapshot.update(congestion_feats)

    # STRICT LEAKAGE GUARDS: Verify target fields are not present
    prohibited_keys = (
        "outtime",
        "disposition",
        "remaining_time_minutes",
        "remaining_time",
    )
    for key in prohibited_keys:
        if key in snapshot:
            del snapshot[key]

    return snapshot


def generate_snapshots_dataset(
    edstays_df: pd.DataFrame,
    triage_df: Optional[pd.DataFrame] = None,
    vitalsign_df: Optional[pd.DataFrame] = None,
    snapshot_offsets_minutes: Sequence[int] = (0, 30, 60),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate batch training snapshots and strictly separated target labels.

    For each encounter, generates snapshots at (intime + offset) for all offsets
    that occur strictly before outtime.

    Args:
        edstays_df: Cleaned edstays encounters DataFrame.
        triage_df: Optional cleaned triage DataFrame.
        vitalsign_df: Optional cleaned vitalsign DataFrame.
        snapshot_offsets_minutes: Offsets in minutes after arrival to generate
            snapshots.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: DataFrame of prediction features (never contains target fields)
            - y: Series of target labels (remaining_time_minutes) aligned with X.
    """
    features_list: List[Dict[str, Any]] = []
    labels_list: List[float] = []

    for _, row in edstays_df.iterrows():
        stay_id = int(row["stay_id"])
        intime = pd.to_datetime(row["intime"])
        outtime = pd.to_datetime(row["outtime"])

        for offset in snapshot_offsets_minutes:
            snap_time = intime + timedelta(minutes=offset)

            # Snapshots can only be taken while the patient is still in the department
            if snap_time >= outtime:
                continue

            try:
                feat_dict = create_patient_snapshot(
                    stay_id=stay_id,
                    snapshot_time=snap_time,
                    edstays_df=edstays_df,
                    triage_df=triage_df,
                    vitalsign_df=vitalsign_df,
                )
                label = create_remaining_time_label(outtime, snap_time)

                features_list.append(feat_dict)
                labels_list.append(label)
            except Exception as exc:
                logger.warning(
                    "Skipping snapshot for stay_id %d at offset %d: %s",
                    stay_id,
                    offset,
                    exc,
                )

    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list, name="remaining_time_minutes")
    return X, y
