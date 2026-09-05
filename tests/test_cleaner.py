"""
Unit tests for QueueMind data cleaning and physiological plausibility modules.
"""

import numpy as np
import pandas as pd

from queuemind.data.cleaner import (
    clean_edstays,
    clean_triage,
    clean_vitalsign,
    filter_physiological_vitals,
    parse_pain_score,
)


def test_parse_pain_score() -> None:
    """Test parse_pain_score on diverse recording styles."""
    assert parse_pain_score("0") == 0.0
    assert parse_pain_score("5") == 5.0
    assert parse_pain_score(7) == 7.0
    assert parse_pain_score("10") == 10.0
    assert parse_pain_score("mild") == 2.0
    assert parse_pain_score("moderate") == 5.0
    assert parse_pain_score("severe") == 8.0
    assert parse_pain_score("unable to assess") is None
    assert parse_pain_score("uta") is None
    assert parse_pain_score(None) is None
    assert parse_pain_score(np.nan) is None
    assert parse_pain_score("denies") == 0.0
    # Range handling - extracts first number
    assert parse_pain_score("6-7") == 6.0


def test_filter_physiological_vitals() -> None:
    """Test physiological limits nullify impossible readings while keeping plausible."""
    df = pd.DataFrame(
        {
            "heartrate": [15.0, 72.0, 185.0, 320.0],  # 15 and 320 are impossible
            "temperature": [75.0, 98.6, 103.5, 120.0],  # 75 and 120 are impossible
            "sbp": [25.0, 120.0, 210.0, 310.0],  # 25 and 310 are impossible
            "dbp": [10.0, 80.0, 120.0, 180.0],  # 10 and 180 are impossible
            "resprate": [2.0, 16.0, 42.0, 85.0],  # 2 and 85 are impossible
            "o2sat": [30.0, 98.0, 91.0, 105.0],  # 30 and 105 are impossible
        }
    )

    filtered = filter_physiological_vitals(df)

    # Plausible values are preserved
    assert filtered.loc[1, "heartrate"] == 72.0
    assert filtered.loc[2, "heartrate"] == 185.0
    assert filtered.loc[1, "temperature"] == 98.6
    assert filtered.loc[2, "temperature"] == 103.5

    # Impossible values are nullified
    assert pd.isna(filtered.loc[0, "heartrate"])
    assert pd.isna(filtered.loc[3, "heartrate"])
    assert pd.isna(filtered.loc[0, "temperature"])
    assert pd.isna(filtered.loc[3, "temperature"])
    assert pd.isna(filtered.loc[0, "sbp"])
    assert pd.isna(filtered.loc[3, "sbp"])
    assert pd.isna(filtered.loc[0, "dbp"])
    assert pd.isna(filtered.loc[3, "dbp"])
    assert pd.isna(filtered.loc[0, "resprate"])
    assert pd.isna(filtered.loc[3, "resprate"])
    assert pd.isna(filtered.loc[0, "o2sat"])
    assert pd.isna(filtered.loc[3, "o2sat"])


def test_filter_inverted_blood_pressure() -> None:
    """Test that SBP < DBP paradox is detected and nullified."""
    df = pd.DataFrame(
        {
            "sbp": [120.0, 70.0],
            "dbp": [80.0, 110.0],  # Row 1 has SBP (70) < DBP (110)
        }
    )
    filtered = filter_physiological_vitals(df)
    assert filtered.loc[0, "sbp"] == 120.0
    assert filtered.loc[0, "dbp"] == 80.0
    assert pd.isna(filtered.loc[1, "sbp"])
    assert pd.isna(filtered.loc[1, "dbp"])


def test_clean_edstays() -> None:
    """Test cleaning of edstays master table."""
    df = pd.DataFrame(
        {
            "STAY_ID": [100, 101, 102, 100],  # Contains duplicate 100
            "INTIME": [
                "2026-01-01 10:00:00",
                "2026-01-01 11:00:00",
                "2026-01-01 15:00:00",
                "2026-01-01 10:00:00",
            ],
            "OUTTIME": [
                "2026-01-01 14:00:00",
                "2026-01-01 10:00:00",  # Inverted: outtime < intime
                "2026-01-01 18:00:00",
                "2026-01-01 14:00:00",
            ],
            "gender": ["m ", "f", "M", "m"],
            "arrival_transport": ["WALK IN", "AMBULANCE", "None", "WALK IN"],
        }
    )

    cleaned = clean_edstays(df)

    # Inverted row 101 should be dropped
    # Duplicate row 100 should be deduplicated
    assert len(cleaned) == 2
    assert set(cleaned["stay_id"]) == {100, 102}
    assert cleaned.loc[cleaned["stay_id"] == 100, "gender"].iloc[0] == "M"
    assert pd.isna(cleaned.loc[cleaned["stay_id"] == 102, "arrival_transport"].iloc[0])


def test_clean_triage() -> None:
    """Test cleaning of triage table."""
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "acuity": [2, 0, 7],  # 0 and 7 are out of bounds (valid: 1-5)
            "heartrate": [80.0, 350.0, 90.0],  # 350 is impossible
            "pain": ["5", "severe", "uta"],
            "chiefcomplaint": [" Chest Pain ", "FEVER", "nan"],
        }
    )

    cleaned = clean_triage(df)
    assert cleaned.loc[0, "acuity"] == 2.0
    assert pd.isna(cleaned.loc[1, "acuity"])
    assert pd.isna(cleaned.loc[2, "acuity"])
    assert pd.isna(cleaned.loc[1, "heartrate"])
    assert cleaned.loc[0, "pain"] == 5.0
    assert cleaned.loc[1, "pain"] == 8.0
    assert pd.isna(cleaned.loc[2, "pain"])
    assert cleaned.loc[0, "chiefcomplaint"] == "chest pain"
    assert pd.isna(cleaned.loc[2, "chiefcomplaint"])


def test_clean_vitalsign() -> None:
    """Test cleaning of longitudinal vitalsign table."""
    df = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "charttime": [
                "2026-01-01 10:15:00",
                "2026-01-01 10:15:00",  # Duplicate timestamp
                "not_a_time",  # Unparseable
            ],
            "heartrate": [85.0, 85.0, 90.0],
            "o2sat": [98.0, 98.0, 95.0],
        }
    )

    cleaned = clean_vitalsign(df)
    # Deduplicated timestamp and dropped invalid time
    assert len(cleaned) == 1
    assert cleaned.loc[0, "stay_id"] == 1
    assert cleaned.loc[0, "heartrate"] == 85.0
