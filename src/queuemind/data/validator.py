"""
Data validation utilities for QueueMind datasets.

Provides modular validation routines to ensure schema integrity, absence of
duplicate keys, temporal coherence, non-emptiness, and missing value detection
for raw and transformed emergency department data.
"""

import logging
from typing import Any, Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Exception raised when a dataset fails critical validation constraints."""

    pass


def check_non_empty(df: pd.DataFrame, table_name: str = "DataFrame") -> None:
    """
    Verify that a DataFrame contains at least one row.

    Args:
        df: The pandas DataFrame to check.
        table_name: Human-readable name of the table for log messages.

    Raises:
        DataValidationError: If the DataFrame is empty.
    """
    if df.empty:
        msg = f"Validation failed: {table_name} is empty (0 rows)."
        logger.error(msg)
        raise DataValidationError(msg)


def check_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str = "DataFrame",
) -> None:
    """
    Verify that all specified required columns are present in the DataFrame.

    Args:
        df: The pandas DataFrame to inspect.
        required_columns: Iterable of expected column names.
        table_name: Human-readable name of the table.

    Raises:
        DataValidationError: If any required column is missing.
    """
    missing = sorted(list(set(required_columns) - set(df.columns)))
    if missing:
        msg = f"Validation failed for {table_name}: Missing required columns: {missing}"
        logger.error(msg)
        raise DataValidationError(msg)


def check_duplicate_identifiers(
    df: pd.DataFrame,
    id_column: str,
    table_name: str = "DataFrame",
) -> None:
    """
    Verify that an identifier column contains no duplicate entries.

    Args:
        df: The pandas DataFrame to inspect.
        id_column: Name of the identifier column (e.g., 'stay_id').
        table_name: Human-readable name of the table.

    Raises:
        DataValidationError: If duplicate identifier values are detected.
    """
    if id_column not in df.columns:
        msg = (
            f"Validation failed for {table_name}: ID column '{id_column}' "
            f"not found in DataFrame."
        )
        logger.error(msg)
        raise DataValidationError(msg)

    num_duplicates = int(df[id_column].duplicated().sum())
    if num_duplicates > 0:
        msg = (
            f"Validation failed for {table_name}: Found {num_duplicates} "
            f"duplicate values in primary key '{id_column}'."
        )
        logger.error(msg)
        raise DataValidationError(msg)


def check_datetime_columns(
    df: pd.DataFrame,
    datetime_columns: Iterable[str],
    table_name: str = "DataFrame",
) -> None:
    """
    Verify that designated datetime columns can be parsed into valid timestamps.

    Args:
        df: The pandas DataFrame to inspect.
        datetime_columns: Iterable of timestamp column names.
        table_name: Human-readable name of the table.

    Raises:
        DataValidationError: If any column fails timestamp conversion.
    """
    for col in datetime_columns:
        if col not in df.columns:
            continue
        try:
            pd.to_datetime(df[col], errors="raise")
        except Exception as exc:
            msg = (
                f"Validation failed for {table_name}: Column '{col}' contains "
                f"unparseable datetime values: {exc}"
            )
            logger.error(msg)
            raise DataValidationError(msg) from exc


def check_missing_values(
    df: pd.DataFrame,
    critical_columns: Optional[Iterable[str]] = None,
    table_name: str = "DataFrame",
) -> Dict[str, int]:
    """
    Calculate missing value counts and raise errors for critical columns.

    Args:
        df: The pandas DataFrame to inspect.
        critical_columns: Optional columns that must not contain missing values.
        table_name: Human-readable name of the table.

    Returns:
        Dict[str, int]: Mapping of column name to missing row count.

    Raises:
        DataValidationError: If any critical column contains null values.
    """
    missing_counts = {
        col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0
    }

    if critical_columns:
        for col in critical_columns:
            if col in missing_counts and missing_counts[col] > 0:
                msg = (
                    f"Validation failed for {table_name}: Critical column '{col}' "
                    f"contains {missing_counts[col]} missing values."
                )
                logger.error(msg)
                raise DataValidationError(msg)

    return missing_counts


def check_time_ordering(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    table_name: str = "DataFrame",
) -> int:
    """
    Verify that start timestamps precede or equal end timestamps.

    Args:
        df: The pandas DataFrame to inspect.
        start_col: Column name representing the beginning timestamp (e.g. 'intime').
        end_col: Column name representing the conclusion timestamp (e.g. 'outtime').
        table_name: Human-readable name of the table.

    Returns:
        int: Number of invalid records found (0 if completely valid).

    Raises:
        DataValidationError: If end timestamps strictly precede start timestamps.
    """
    check_required_columns(df, [start_col, end_col], table_name=table_name)
    start_dt = pd.to_datetime(df[start_col])
    end_dt = pd.to_datetime(df[end_col])

    invalid_mask = end_dt < start_dt
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        msg = (
            f"Validation failed for {table_name}: Found {invalid_count} records "
            f"where '{end_col}' precedes '{start_col}'."
        )
        logger.error(msg)
        raise DataValidationError(msg)

    return 0


def validate_ed_stays(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the core emergency department encounter table (edstays).

    Enforces:
    - Non-empty DataFrame
    - Presence of required columns:
      ('stay_id', 'subject_id', 'intime', 'outtime', 'acuity')
    - Uniqueness of 'stay_id'
    - Parseability of 'intime' and 'outtime'
    - Temporal coherence (outtime >= intime)
    - Warning on missing acuity ratings

    Args:
        df: The edstays DataFrame to validate.

    Returns:
        Dict[str, Any]: Validation summary containing 'is_valid',
            'errors', and 'warnings'.

    Raises:
        DataValidationError: If critical validation checks fail.
    """
    report: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}
    table_name = "edstays"

    try:
        check_non_empty(df, table_name=table_name)
    except DataValidationError as e:
        report["is_valid"] = False
        report["errors"].append(str(e))
        raise

    required_cols = ["stay_id", "subject_id", "intime", "outtime", "acuity"]
    try:
        check_required_columns(df, required_cols, table_name=table_name)
    except DataValidationError as e:
        report["is_valid"] = False
        report["errors"].append(str(e))
        raise

    # Check stay_id uniqueness
    try:
        check_duplicate_identifiers(df, "stay_id", table_name=table_name)
    except DataValidationError as e:
        report["is_valid"] = False
        report["errors"].append(str(e))

    # Check datetime validity and chronological ordering
    try:
        check_datetime_columns(df, ["intime", "outtime"], table_name=table_name)
        check_time_ordering(df, "intime", "outtime", table_name=table_name)
    except DataValidationError as e:
        report["is_valid"] = False
        report["errors"].append(str(e))

    # Missing acuity is treated as an operational warning
    missing_acuity = int(df["acuity"].isna().sum())
    if missing_acuity > 0:
        warn_msg = f"Found {missing_acuity} records with missing acuity"
        report["warnings"].append(warn_msg)
        logger.warning(warn_msg)

    if not report["is_valid"]:
        raise DataValidationError(f"Data validation failed: {report['errors']}")

    logger.info("ED Stays dataset validation passed successfully.")
    return report
