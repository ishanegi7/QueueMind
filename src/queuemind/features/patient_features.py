"""
Patient-level feature extraction with strict point-in-time constraints.

Extracts patient demographics, triage assessment, point-in-time longitudinal vitals,
and clinical abnormality flags available strictly at or before the prediction snapshot.
"""

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard ED chief complaint keyword mapping into broad clinical categories
CHIEF_COMPLAINT_CATEGORIES: Dict[str, Sequence[str]] = {
    "chest_pain": ("chest", "angina", "cardiac", "heart"),
    "abdominal_pain": ("abd", "stomach", "nausea", "vomit", "gi"),
    "respiratory": ("breath", "sob", "dyspnea", "asthma", "cough", "wheeze"),
    "neurological": ("headache", "dizzy", "stroke", "seizure", "syncope", "numb"),
    "trauma_injury": (
        "fall",
        "trauma",
        "injury",
        "fracture",
        "mvc",
        "accident",
        "wound",
        "laceration",
    ),
    "fever_infection": ("fever", "chills", "infection", "sepsis", "cellulitis"),
    "psychiatric": ("psych", "depression", "suicid", "anxiety", "hallucination"),
}


def categorize_chief_complaint(complaint: Optional[str]) -> str:
    """
    Map raw chief complaint text to a standard clinical category.

    Args:
        complaint: Raw chief complaint string.

    Returns:
        str: Mapped category name or 'other' / 'missing'.
    """
    if complaint is None or pd.isna(complaint):
        return "missing"

    clean_text = str(complaint).lower().strip()
    if not clean_text or clean_text in ("none", "nan"):
        return "missing"

    for category, keywords in CHIEF_COMPLAINT_CATEGORIES.items():
        if any(keyword in clean_text for keyword in keywords):
            return category

    return "other"


def extract_patient_features(
    stay_id: int,
    snapshot_time: pd.Timestamp,
    intime: pd.Timestamp,
    triage_row: Optional[pd.Series] = None,
    vitals_history: Optional[pd.DataFrame] = None,
    arrival_transport: Optional[str] = None,
    gender: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract point-in-time patient features at snapshot_time.

    CRITICAL LEAKAGE GUARANTEE:
    - Only vitals recorded at charttime <= snapshot_time are evaluated.
    - Future vitals (charttime > snapshot_time) are strictly filtered out.
    - outtime and disposition are strictly excluded.

    Args:
        stay_id: Emergency department encounter identifier.
        snapshot_time: Hard information cutoff timestamp.
        intime: Encounter presentation timestamp.
        triage_row: Optional Series from triage assessment conducted at admission.
        vitals_history: Optional DataFrame of periodic vitals for this patient.
        arrival_transport: Mode of arrival (e.g. 'AMBULANCE', 'WALK IN').
        gender: Patient biological sex.

    Returns:
        Dict[str, Any]: Dictionary of point-in-time patient features.

    Raises:
        ValueError: If snapshot_time is strictly before intime.
    """
    snap_dt = pd.to_datetime(snapshot_time)
    in_dt = pd.to_datetime(intime)

    if snap_dt < in_dt:
        msg = (
            f"Snapshot time ({snap_dt}) precedes patient arrival intime ({in_dt}) "
            f"for stay_id {stay_id}."
        )
        logger.error(msg)
        raise ValueError(msg)

    elapsed_minutes = (snap_dt - in_dt).total_seconds() / 60.0

    features: Dict[str, Any] = {
        "stay_id": stay_id,
        "elapsed_time_minutes": elapsed_minutes,
        "gender": str(gender).upper() if gender and pd.notna(gender) else "UNKNOWN",
        "arrival_transport": (
            str(arrival_transport).upper()
            if arrival_transport and pd.notna(arrival_transport)
            else "UNKNOWN"
        ),
    }

    # Extract triage assessment features (assessed at arrival <= snapshot_time)
    if triage_row is not None and not triage_row.empty:
        features["acuity"] = (
            float(triage_row["acuity"])
            if "acuity" in triage_row and pd.notna(triage_row["acuity"])
            else np.nan
        )
        features["triage_pain"] = (
            float(triage_row["pain"])
            if "pain" in triage_row and pd.notna(triage_row["pain"])
            else np.nan
        )
        complaint = (
            triage_row["chiefcomplaint"] if "chiefcomplaint" in triage_row else None
        )
        features["chiefcomplaint_category"] = categorize_chief_complaint(complaint)

        for vital in ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"):
            features[f"triage_{vital}"] = (
                float(triage_row[vital])
                if vital in triage_row and pd.notna(triage_row[vital])
                else np.nan
            )
    else:
        features["acuity"] = np.nan
        features["triage_pain"] = np.nan
        features["chiefcomplaint_category"] = "missing"
        for vital in ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"):
            features[f"triage_{vital}"] = np.nan

    # Process longitudinal vitals strictly up to snapshot_time
    num_vitals = 0
    last_vitals: Dict[str, float] = {}

    if vitals_history is not None and not vitals_history.empty:
        # POINT-IN-TIME FILTER: Strictly charttime <= snapshot_time
        time_mask = pd.to_datetime(vitals_history["charttime"]) <= snap_dt
        if "stay_id" in vitals_history.columns:
            valid_vitals = vitals_history[
                time_mask & (vitals_history["stay_id"] == stay_id)
            ].copy()
        else:
            valid_vitals = vitals_history[time_mask].copy()

        num_vitals = len(valid_vitals)

        if num_vitals > 0:
            valid_vitals = valid_vitals.sort_values("charttime")
            latest_row = valid_vitals.iloc[-1]

            for v in ("heartrate", "sbp", "dbp", "o2sat", "resprate", "temperature"):
                if v in latest_row and pd.notna(latest_row[v]):
                    last_vitals[v] = float(latest_row[v])

            # Rolling vital statistics up to snapshot
            for v in ("heartrate", "sbp", "o2sat"):
                if v in valid_vitals.columns:
                    valid_series = pd.to_numeric(
                        valid_vitals[v], errors="coerce"
                    ).dropna()
                    if not valid_series.empty:
                        features[f"mean_{v}"] = float(valid_series.mean())
                        features[f"min_{v}"] = float(valid_series.min())
                        features[f"max_{v}"] = float(valid_series.max())

    features["num_vital_measurements"] = num_vitals

    # Populate latest available vitals
    # (falls back to triage vitals if none prior to snapshot)
    for v in ("heartrate", "sbp", "dbp", "o2sat", "resprate", "temperature"):
        if v in last_vitals:
            features[f"last_{v}"] = last_vitals[v]
        elif pd.notna(features.get(f"triage_{v}")):
            features[f"last_{v}"] = features[f"triage_{v}"]
        else:
            features[f"last_{v}"] = np.nan

    # Clinical abnormality indicators based on latest available vitals
    last_hr = features.get("last_heartrate")
    if last_hr is not None and pd.notna(last_hr):
        hr_val = float(last_hr)
        features["is_tachycardic"] = int(hr_val > 100.0)
        features["is_bradycardic"] = int(hr_val < 60.0)
    else:
        features["is_tachycardic"] = np.nan
        features["is_bradycardic"] = np.nan

    last_o2 = features.get("last_o2sat")
    if last_o2 is not None and pd.notna(last_o2):
        features["is_hypoxic"] = int(float(last_o2) < 92.0)
    else:
        features["is_hypoxic"] = np.nan

    last_sbp = features.get("last_sbp")
    if last_sbp is not None and pd.notna(last_sbp):
        sbp_val = float(last_sbp)
        features["is_hypotensive"] = int(sbp_val < 90.0)
        features["is_hypertensive_crisis"] = int(sbp_val >= 180.0)
    else:
        features["is_hypotensive"] = np.nan
        features["is_hypertensive_crisis"] = np.nan

    last_temp = features.get("last_temperature")
    if last_temp is not None and pd.notna(last_temp):
        features["is_febrile"] = int(float(last_temp) >= 100.4)
    else:
        features["is_febrile"] = np.nan

    return features
