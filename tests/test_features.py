"""
Unit tests for QueueMind feature engineering and point-in-time snapshot generation.

Includes strict data leakage invariance tests to guarantee that future events
cannot contaminate features at snapshot time T.
"""

import numpy as np
import pandas as pd
import pytest

from queuemind.features.congestion_features import calculate_congestion_features
from queuemind.features.patient_features import (
    categorize_chief_complaint,
    extract_patient_features,
)
from queuemind.features.snapshots import (
    create_patient_snapshot,
    create_remaining_time_label,
    generate_snapshots_dataset,
)
from queuemind.features.temporal_features import (
    extract_temporal_features,
    get_shift_bucket,
)

# ==============================================================================
# Patient Features Tests
# ==============================================================================


def test_categorize_chief_complaint() -> None:
    """Test chief complaint text mapping to standard clinical categories."""
    assert (
        categorize_chief_complaint("Chest pain with shortness of breath")
        == "chest_pain"
    )
    assert categorize_chief_complaint("Severe abdominal cramps") == "abdominal_pain"
    assert categorize_chief_complaint("Difficulty breathing / asthma") == "respiratory"
    assert categorize_chief_complaint("Fall from ladder, laceration") == "trauma_injury"
    assert categorize_chief_complaint("Headache and dizziness") == "neurological"
    assert categorize_chief_complaint("High fever and chills") == "fever_infection"
    assert categorize_chief_complaint("Depression and anxiety") == "psychiatric"
    assert categorize_chief_complaint("Skin rash on arm") == "other"
    assert categorize_chief_complaint(None) == "missing"
    assert categorize_chief_complaint("") == "missing"


def test_extract_patient_features_point_in_time() -> None:
    """Test that longitudinal vitals strictly after snapshot_time are excluded."""
    stay_id = 501
    intime = pd.Timestamp("2026-01-01 10:00:00")
    snapshot_time = pd.Timestamp("2026-01-01 11:30:00")

    triage_row = pd.Series(
        {
            "acuity": 2.0,
            "pain": 7.0,
            "chiefcomplaint": "Chest pain",
            "heartrate": 88.0,
            "sbp": 130.0,
            "dbp": 82.0,
            "o2sat": 98.0,
            "temperature": 98.6,
            "resprate": 18.0,
        }
    )

    vitals_history = pd.DataFrame(
        {
            "stay_id": [stay_id, stay_id, stay_id],
            "charttime": [
                "2026-01-01 10:30:00",  # Prior to snapshot (valid)
                "2026-01-01 11:15:00",  # Prior to snapshot (valid - latest)
                "2026-01-01 12:00:00",  # AFTER snapshot (must be excluded)
            ],
            "heartrate": [92.0, 105.0, 140.0],  # 140 is in the future
            "sbp": [132.0, 138.0, 80.0],  # 80 is in the future
            "o2sat": [97.0, 96.0, 88.0],
        }
    )

    feats = extract_patient_features(
        stay_id=stay_id,
        snapshot_time=snapshot_time,
        intime=intime,
        triage_row=triage_row,
        vitals_history=vitals_history,
        arrival_transport="AMBULANCE",
        gender="M",
    )

    # Check elapsed time
    assert feats["elapsed_time_minutes"] == 90.0

    # Num vitals up to snapshot must be 2, NOT 3
    assert feats["num_vital_measurements"] == 2

    # Latest vitals must be from 11:15 (HR=105, SBP=138), NOT 12:00 (HR=140, SBP=80)
    assert feats["last_heartrate"] == 105.0
    assert feats["last_sbp"] == 138.0

    # Clinical flags
    assert feats["is_tachycardic"] == 1  # 105 > 100
    assert feats["is_bradycardic"] == 0
    assert feats["is_hypotensive"] == 0  # 138 is not < 90


def test_extract_patient_features_preceding_intime_raises() -> None:
    """Test error when snapshot_time is before patient arrival intime."""
    with pytest.raises(ValueError) as exc:
        extract_patient_features(
            stay_id=1,
            snapshot_time=pd.Timestamp("2026-01-01 09:00:00"),
            intime=pd.Timestamp("2026-01-01 10:00:00"),
        )
    assert "precedes" in str(exc.value)


# ==============================================================================
# Temporal Features Tests
# ==============================================================================


def test_get_shift_bucket() -> None:
    """Test shift bucket determination."""
    assert get_shift_bucket(8) == "morning"
    assert get_shift_bucket(14) == "morning"
    assert get_shift_bucket(15) == "evening"
    assert get_shift_bucket(22) == "evening"
    assert get_shift_bucket(23) == "night"
    assert get_shift_bucket(4) == "night"


def test_extract_temporal_features() -> None:
    """Test temporal feature extraction and cyclical encodings."""
    snap_dt = pd.Timestamp("2026-01-03 14:00:00")  # Saturday
    arr_dt = pd.Timestamp("2026-01-03 12:00:00")

    feats = extract_temporal_features(snap_dt, arrival_time=arr_dt)

    assert feats["snapshot_hour"] == 14
    assert feats["snapshot_day_of_week"] == 5  # Saturday
    assert feats["snapshot_is_weekend"] == 1
    assert feats["snapshot_shift"] == "morning"
    assert feats["elapsed_time_minutes"] == 120.0
    assert feats["arrival_hour"] == 12

    # Cyclical sin/cos consistency: sin^2 + cos^2 = 1.0
    sin_val = feats["snapshot_hour_sin"]
    cos_val = feats["snapshot_hour_cos"]
    assert np.isclose(sin_val**2 + cos_val**2, 1.0, atol=1e-5)


# ==============================================================================
# Congestion Features Tests
# ==============================================================================


def test_calculate_congestion_features() -> None:
    """Test active census and rolling arrival/discharge calculation."""
    # Create encounter history:
    # Patient 1: in 10:00, out 12:00 (active at 11:30)
    # Patient 2: in 11:00, out 14:00 (active at 11:30, arrived within 30m)
    # Patient 3: in 11:20, out 15:00 (active at 11:30, arrived within 15m)
    # Patient 4: in 09:00, out 11:10 (departed at 11:10, within 30m before snapshot)
    # Patient 5: in 11:45, out 16:00 (arrived AFTER snapshot - future event!)
    edstays_df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "intime": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 11:20:00",
                "2026-01-01 09:00:00",
                "2026-01-01 11:45:00",
            ],
            "outtime": [
                "2026-01-01 12:00:00",
                "2026-01-01 14:00:00",
                "2026-01-01 15:00:00",
                "2026-01-01 11:10:00",
                "2026-01-01 16:00:00",
            ],
        }
    )

    triage_df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "acuity": [
                3.0,
                2.0,
                1.0,
                4.0,
                2.0,
            ],  # Patients 2 & 3 are high acuity (<= 2)
        }
    )

    snap_time = pd.Timestamp("2026-01-01 11:30:00")
    cong = calculate_congestion_features(snap_time, edstays_df, triage_df)

    # Active patients at 11:30: Patients 1, 2, 3
    # (Patient 4 departed, Patient 5 not arrived yet)
    assert cong["active_patient_count"] == 3

    # Arrivals in (11:15, 11:30]: Patient 3 (at 11:20) -> 1 arrival
    assert cong["recent_arrivals_15m"] == 1
    # Arrivals in (11:00, 11:30]: Patients 2, 3 -> 2 arrivals
    assert cong["recent_arrivals_30m"] == 2
    # Arrivals in (10:30, 11:30]: Patients 2, 3 -> 2 arrivals
    assert cong["recent_arrivals_60m"] == 2

    # Departures in (11:00, 11:30]: Patient 4 departed at 11:10 -> 1 departure
    assert cong["recent_departures_30m"] == 1

    # High acuity among active (Patients 2 and 3)
    assert cong["high_acuity_active_count"] == 2


# ==============================================================================
# Snapshot Generator & Target Preparation Tests
# ==============================================================================


def test_create_remaining_time_label() -> None:
    """Test target label calculation in minutes."""
    outtime = pd.Timestamp("2026-01-01 14:00:00")
    snap_time = pd.Timestamp("2026-01-01 11:30:00")

    label = create_remaining_time_label(outtime, snap_time)
    assert label == 150.0  # 2.5 hours = 150 minutes

    # Error if snapshot is after outtime
    with pytest.raises(ValueError):
        create_remaining_time_label(
            pd.Timestamp("2026-01-01 11:00:00"),
            pd.Timestamp("2026-01-01 12:00:00"),
        )


def test_create_patient_snapshot_prohibits_targets() -> None:
    """Test that patient snapshot NEVER contains target columns."""
    edstays_df = pd.DataFrame(
        {
            "stay_id": [10],
            "intime": ["2026-01-01 10:00:00"],
            "outtime": ["2026-01-01 14:00:00"],
            "disposition": ["ADMITTED"],
            "gender": ["F"],
            "arrival_transport": ["WALK IN"],
        }
    )
    snap_time = pd.Timestamp("2026-01-01 11:00:00")
    snapshot = create_patient_snapshot(10, snap_time, edstays_df)

    # Prohibited target columns must NOT exist in features
    assert "outtime" not in snapshot
    assert "disposition" not in snapshot
    assert "remaining_time_minutes" not in snapshot
    assert "remaining_time" not in snapshot
    assert snapshot["elapsed_time_minutes"] == 60.0


def test_generate_snapshots_dataset() -> None:
    """Test batch snapshot generation producing separated X and y."""
    edstays_df = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "intime": ["2026-01-01 10:00:00", "2026-01-01 10:30:00"],
            "outtime": ["2026-01-01 12:00:00", "2026-01-01 11:15:00"],
            "gender": ["M", "F"],
            "arrival_transport": ["WALK IN", "AMBULANCE"],
        }
    )

    X, y = generate_snapshots_dataset(edstays_df, snapshot_offsets_minutes=(0, 30, 60))

    assert len(X) == len(y)
    assert not X.empty
    assert "outtime" not in X.columns
    assert "disposition" not in X.columns
    assert y.name == "remaining_time_minutes"
    # All remaining times must be positive
    assert (y > 0).all()


# ==============================================================================
# CRITICAL LEAKAGE INVARIANCE TEST
# ==============================================================================


def test_future_event_leakage_invariance() -> None:
    """
    STRICT VERIFICATION: Future events must not alter past snapshot features.

    Creates a base world at T = 12:00:00.
    Then introduces various future perturbations (future vitals, future arrivals,
    future departures) occurring at T > 12:00:00.
    Asserts that the extracted features at T = 12:00:00 are 100% IDENTICAL.
    """
    target_stay_id = 100
    snap_time = pd.Timestamp("2026-01-01 12:00:00")

    # Base world
    base_edstays = pd.DataFrame(
        {
            "stay_id": [100, 101, 102],
            "intime": [
                "2026-01-01 10:00:00",  # Target patient
                "2026-01-01 11:00:00",  # Active co-patient
                "2026-01-01 09:00:00",  # Historical departed patient
            ],
            "outtime": [
                "2026-01-01 15:00:00",  # Target departs at 15:00
                "2026-01-01 14:00:00",  # Co-patient departs at 14:00
                "2026-01-01 11:30:00",  # Historical departs before snap
            ],
            "gender": ["M", "F", "M"],
            "arrival_transport": ["AMBULANCE", "WALK IN", "WALK IN"],
        }
    )

    base_triage = pd.DataFrame(
        {
            "stay_id": [100, 101, 102],
            "acuity": [2.0, 3.0, 4.0],
            "heartrate": [90.0, 80.0, 75.0],
            "sbp": [130.0, 120.0, 115.0],
        }
    )

    base_vitals = pd.DataFrame(
        {
            "stay_id": [100, 100],
            "charttime": [
                "2026-01-01 10:30:00",
                "2026-01-01 11:30:00",
            ],
            "heartrate": [92.0, 95.0],
            "sbp": [132.0, 135.0],
        }
    )

    # 1. Snapshot in the base world
    base_snapshot = create_patient_snapshot(
        stay_id=target_stay_id,
        snapshot_time=snap_time,
        edstays_df=base_edstays,
        triage_df=base_triage,
        vitalsign_df=base_vitals,
    )

    # 2. Perturbed future world:
    # - Add future vital signs for target patient at 12:45 and 13:30 (extreme readings!)
    perturbed_vitals = pd.concat(
        [
            base_vitals,
            pd.DataFrame(
                {
                    "stay_id": [100, 100],
                    "charttime": [
                        "2026-01-01 12:45:00",  # Future!
                        "2026-01-01 13:30:00",  # Future!
                    ],
                    "heartrate": [180.0, 190.0],  # Severe tachycardia in future
                    "sbp": [60.0, 50.0],  # Severe shock in future
                }
            ),
        ],
        ignore_index=True,
    )

    # - Add future patient arriving at 12:15
    # - Change co-patient 101's departure from 14:00 to 18:00 (both in future)
    perturbed_edstays = pd.DataFrame(
        {
            "stay_id": [100, 101, 102, 103],
            "intime": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 09:00:00",
                "2026-01-01 12:15:00",  # Arrived after snapshot
            ],
            "outtime": [
                "2026-01-01 16:00:00",  # Changed future outtime
                "2026-01-01 18:00:00",  # Changed future outtime
                "2026-01-01 11:30:00",
                "2026-01-01 19:00:00",
            ],
            "gender": ["M", "F", "M", "F"],
            "arrival_transport": ["AMBULANCE", "WALK IN", "WALK IN", "AMBULANCE"],
        }
    )

    perturbed_snapshot = create_patient_snapshot(
        stay_id=target_stay_id,
        snapshot_time=snap_time,
        edstays_df=perturbed_edstays,
        triage_df=base_triage,
        vitalsign_df=perturbed_vitals,
    )

    # 3. VERIFY 100% INVARIANCE
    assert base_snapshot == perturbed_snapshot, (
        f"Leakage detected! Features differ despite only future events changing:\n"
        f"Diff: {set(base_snapshot.items()) ^ set(perturbed_snapshot.items())}"
    )
