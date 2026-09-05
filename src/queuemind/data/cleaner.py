"""
Data cleaning and physiological plausibility filtering for MIMIC-IV-ED tables.

Provides non-destructive cleaning routines that:
- Normalize column names and string values
- Parse timestamps to timezone-naive datetimes
- Handle missing data and deduplicate identifiers
- Distinguish between impossible measurement artifacts and extreme clinical states
- Follow documented physiological thresholds based on clinical emergency literature
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Documented physiological limits for emergency department vital signs.
# Values strictly outside [min, max] represent device disconnects, typographical
# errors, or corrupted readings and are nullified to NaN rather than dropping
# the entire encounter.
PHYSIOLOGICAL_RANGES: Dict[str, Tuple[float, float]] = {
    # Temperature in Fahrenheit: human survival limits ~75°F to ~115°F.
    # Standard clinical plausible bounds: 85.0°F to 108.0°F.
    "temperature": (85.0, 108.0),
    # Heart rate in beats per minute: extreme cardiac limits 20 to 260 bpm.
    "heartrate": (20.0, 260.0),
    # Respiratory rate in breaths per minute: 4 to 60 breaths/min.
    "resprate": (4.0, 60.0),
    # Pulse oximetry oxygen saturation (percentage): 50% to 100%.
    "o2sat": (50.0, 100.0),
    # Systolic blood pressure in mmHg: 40 to 260 mmHg.
    "sbp": (40.0, 260.0),
    # Diastolic blood pressure in mmHg: 20 to 160 mmHg.
    "dbp": (20.0, 160.0),
}


def parse_pain_score(value: Any) -> Optional[float]:
    """
    Parse heterogeneously recorded pain scores into numeric values in [0.0, 10.0].

    MIMIC-IV-ED records pain as numeric strings, ranges ('5-6'), qualitative
    descriptors ('critical', 'severe', 'unable to assess'), or missing values.

    Args:
        value: Raw pain score entry.

    Returns:
        Optional[float]: Numeric pain rating in 0.0–10.0, or None if unparseable.
    """
    if pd.isna(value):
        return None

    str_val = str(value).strip().lower()
    if str_val in ("", "none", "unable to assess", "uta", "denies", "no"):
        return 0.0 if str_val in ("none", "denies", "no") else None

    # Handle qualitative heuristics
    if "mild" in str_val:
        return 2.0
    if "moderate" in str_val:
        return 5.0
    if "severe" in str_val:
        return 8.0

    # Extract first valid numeric integer or float
    match = re.search(r"(\d+(?:\.\d+)?)", str_val)
    if match:
        try:
            num = float(match.group(1))
            if 0.0 <= num <= 10.0:
                return num
            if num > 10.0 and num <= 100.0:
                # Scaled 0-100 visual analogue scale
                return num / 10.0
        except ValueError:
            pass

    return None


def filter_physiological_vitals(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Nullify physiologically impossible vital sign values without dropping rows.

    Checks present columns against PHYSIOLOGICAL_RANGES and validates that SBP >= DBP.

    Args:
        df: DataFrame containing vital signs.
        inplace: Whether to modify the DataFrame in place.

    Returns:
        pd.DataFrame: DataFrame with impossible measurements replaced by NaN.
    """
    out_df = df if inplace else df.copy()

    for col, (min_val, max_val) in PHYSIOLOGICAL_RANGES.items():
        if col in out_df.columns:
            # Coerce to numeric
            numeric_col = pd.to_numeric(out_df[col], errors="coerce")
            invalid_mask = (numeric_col < min_val) | (numeric_col > max_val)
            num_invalid = int(invalid_mask.sum())
            if num_invalid > 0:
                logger.info(
                    "Nullified %d impossible values in '%s' outside [%.1f, %.1f]",
                    num_invalid,
                    col,
                    min_val,
                    max_val,
                )
                numeric_col.loc[invalid_mask] = np.nan
            out_df[col] = numeric_col

    # Validate that SBP >= DBP when both are present
    if "sbp" in out_df.columns and "dbp" in out_df.columns:
        inverted_bp = (
            out_df["sbp"].notna()
            & out_df["dbp"].notna()
            & (out_df["sbp"] < out_df["dbp"])
        )
        num_inverted = int(inverted_bp.sum())
        if num_inverted > 0:
            logger.info(
                "Nullified %d records where SBP < DBP (physiological paradox)",
                num_inverted,
            )
            out_df.loc[inverted_bp, ["sbp", "dbp"]] = np.nan

    return out_df


def clean_edstays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and harmonize the master ED encounters table (edstays).

    Steps:
    1. Standardize column names.
    2. Parse intime and outtime into timezone-naive datetimes.
    3. Filter out records where outtime < intime.
    4. Deduplicate by stay_id (keeping the earliest encounter).
    5. Clean string fields (gender, race, arrival_transport, disposition).

    Args:
        df: Raw edstays DataFrame.

    Returns:
        pd.DataFrame: Cleaned edstays dataset.
    """
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    # Convert timestamps
    if "intime" in cleaned.columns:
        cleaned["intime"] = pd.to_datetime(cleaned["intime"], errors="coerce")
    if "outtime" in cleaned.columns:
        cleaned["outtime"] = pd.to_datetime(cleaned["outtime"], errors="coerce")

    # Drop encounters with missing or inverted timestamps
    if "intime" in cleaned.columns and "outtime" in cleaned.columns:
        valid_times = (
            cleaned["intime"].notna()
            & cleaned["outtime"].notna()
            & (cleaned["outtime"] >= cleaned["intime"])
        )
        invalid_count = len(cleaned) - int(valid_times.sum())
        if invalid_count > 0:
            logger.warning(
                "Dropped %d edstays records with missing or inverted timestamps",
                invalid_count,
            )
        cleaned = cleaned[valid_times]

    # Deduplicate stay_id
    if "stay_id" in cleaned.columns:
        dup_count = int(cleaned["stay_id"].duplicated().sum())
        if dup_count > 0:
            logger.warning("Deduplicated %d repeated stay_id entries", dup_count)
            cleaned = cleaned.drop_duplicates(subset=["stay_id"], keep="first")

    # Clean text columns
    text_cols = ["gender", "race", "arrival_transport", "disposition"]
    for col in text_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip().str.upper()
            cleaned[col] = cleaned[col].replace(
                {"NAN": np.nan, "NONE": np.nan, "": np.nan}
            )

    return cleaned.reset_index(drop=True)


def clean_triage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate initial triage assessments.

    Steps:
    1. Standardize column names.
    2. Filter physiological ranges for triage vitals.
    3. Standardize and bounded ESI acuity (1 to 5).
    4. Parse pain scores to numeric float [0.0, 10.0].
    5. Normalize chief complaint strings.

    Args:
        df: Raw triage DataFrame.

    Returns:
        pd.DataFrame: Cleaned triage dataset.
    """
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    # Acuity validation (ESI 1-5)
    if "acuity" in cleaned.columns:
        acuity_num = pd.to_numeric(cleaned["acuity"], errors="coerce")
        invalid_acuity = (acuity_num < 1) | (acuity_num > 5)
        num_invalid = int(invalid_acuity.sum())
        if num_invalid > 0:
            logger.info("Nullified %d out-of-bounds acuity values", num_invalid)
            acuity_num.loc[invalid_acuity] = np.nan
        cleaned["acuity"] = acuity_num

    # Filter physiological bounds on triage vitals
    cleaned = filter_physiological_vitals(cleaned)

    # Standardize pain score
    if "pain" in cleaned.columns:
        cleaned["pain"] = cleaned["pain"].apply(parse_pain_score)

    # Normalize chief complaint
    if "chiefcomplaint" in cleaned.columns:
        cleaned["chiefcomplaint"] = (
            cleaned["chiefcomplaint"].astype(str).str.strip().str.lower()
        )
        cleaned["chiefcomplaint"] = cleaned["chiefcomplaint"].replace(
            {"nan": np.nan, "none": np.nan, "": np.nan}
        )

    # Deduplicate stay_id if present
    if "stay_id" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["stay_id"], keep="first")

    return cleaned.reset_index(drop=True)


def clean_vitalsign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean longitudinal periodic vital signs table.

    Steps:
    1. Standardize column names.
    2. Parse charttime into timezone-naive datetime.
    3. Filter physiological ranges on all periodic vitals.
    4. Standardize pain scores.
    5. Deduplicate exact stay_id + charttime readings.

    Args:
        df: Raw vitalsign DataFrame.

    Returns:
        pd.DataFrame: Cleaned vitalsign dataset.
    """
    cleaned = df.copy()
    cleaned.columns = [col.strip().lower() for col in cleaned.columns]

    # Convert charttime
    if "charttime" in cleaned.columns:
        cleaned["charttime"] = pd.to_datetime(cleaned["charttime"], errors="coerce")
        # Drop rows with unparseable charttime
        missing_time = cleaned["charttime"].isna()
        if missing_time.any():
            logger.warning(
                "Dropped %d vitalsign records with missing charttime",
                int(missing_time.sum()),
            )
            cleaned = cleaned[~missing_time]

    # Filter physiological bounds
    cleaned = filter_physiological_vitals(cleaned)

    # Clean pain
    if "pain" in cleaned.columns:
        cleaned["pain"] = cleaned["pain"].apply(parse_pain_score)

    # Deduplicate stay_id and charttime
    if "stay_id" in cleaned.columns and "charttime" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["stay_id", "charttime"], keep="first")

    return cleaned.reset_index(drop=True)
