"""
Unit tests for QueueMind data loader and validator modules.

All tests utilize synthetic isolated fixtures and pytest temporary directories.
No real or proprietary MIMIC-IV-ED datasets are required or utilized.
"""

import gzip
from pathlib import Path

import pandas as pd
import pytest

from queuemind.data.loader import (
    MIMICDataLoader,
    find_table_file,
    get_data_dir,
    load_mimic_table,
)
from queuemind.data.validator import (
    DataValidationError,
    check_datetime_columns,
    check_duplicate_identifiers,
    check_missing_values,
    check_non_empty,
    check_required_columns,
    check_time_ordering,
    validate_ed_stays,
)

# ==============================================================================
# Loader Unit Tests
# ==============================================================================


def test_get_data_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_data_dir falls back to data/raw if env var is unset."""
    monkeypatch.delenv("MIMIC_DATA_DIR", raising=False)
    assert get_data_dir() == Path("data/raw")


def test_get_data_dir_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test get_data_dir respects MIMIC_DATA_DIR environment variable."""
    custom_dir = tmp_path / "custom_mimic"
    monkeypatch.setenv("MIMIC_DATA_DIR", str(custom_dir))
    assert get_data_dir() == custom_dir


def test_load_mimic_table_missing(tmp_path: Path) -> None:
    """Test loading a non-existent table raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_mimic_table("non_existent_table", data_dir=tmp_path)
    assert "non_existent_table" in str(exc_info.value)


def test_load_mimic_table_csv(tmp_path: Path) -> None:
    """Test loading standard uncompressed CSV table."""
    fixture_df = pd.DataFrame({"stay_id": [1001, 1002], "subject_id": [2001, 2002]})
    csv_file = tmp_path / "edstays.csv"
    fixture_df.to_csv(csv_file, index=False)

    loaded = load_mimic_table("edstays", data_dir=tmp_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["stay_id", "subject_id"]
    assert loaded["stay_id"].tolist() == [1001, 1002]


def test_load_mimic_table_csv_gz(tmp_path: Path) -> None:
    """Test loading gzip-compressed CSV table (.csv.gz)."""
    fixture_df = pd.DataFrame({"stay_id": [1003, 1004], "acuity": [2, 3]})
    gz_file = tmp_path / "triage.csv.gz"
    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        fixture_df.to_csv(f, index=False)

    loaded = load_mimic_table("triage", data_dir=tmp_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["stay_id", "acuity"]
    assert loaded["stay_id"].tolist() == [1003, 1004]


def test_load_mimic_table_parquet(tmp_path: Path) -> None:
    """Test loading Apache Parquet table."""
    fixture_df = pd.DataFrame({"stay_id": [1005, 1006], "heartrate": [80, 95]})
    parquet_file = tmp_path / "vitalsign.parquet"
    fixture_df.to_parquet(parquet_file, index=False)

    loaded = load_mimic_table("vitalsign", data_dir=tmp_path)
    assert len(loaded) == 2
    assert list(loaded.columns) == ["stay_id", "heartrate"]


def test_find_table_file_priority(tmp_path: Path) -> None:
    """Test that parquet has priority over csv when both exist."""
    parquet_file = tmp_path / "edstays.parquet"
    csv_file = tmp_path / "edstays.csv"

    pd.DataFrame({"format": ["parquet"]}).to_parquet(parquet_file, index=False)
    pd.DataFrame({"format": ["csv"]}).to_csv(csv_file, index=False)

    found = find_table_file("edstays", tmp_path)
    assert found is not None
    assert found.suffix == ".parquet"


def test_load_mimic_table_corrupted(tmp_path: Path) -> None:
    """Test that reading a corrupted file raises RuntimeError."""
    corrupted_file = tmp_path / "edstays.parquet"
    corrupted_file.write_bytes(b"corrupted binary data that cannot be parsed")

    with pytest.raises(RuntimeError) as exc_info:
        load_mimic_table("edstays", data_dir=tmp_path)
    assert "Failed to read table" in str(exc_info.value)


def test_mimic_data_loader_class(tmp_path: Path) -> None:
    """Test MIMICDataLoader class methods and table discovery."""
    loader = MIMICDataLoader(data_dir=tmp_path)
    assert loader.get_available_tables() == []

    # Non-existent directory returns empty available list
    non_existent = MIMICDataLoader(data_dir=tmp_path / "missing_dir")
    assert non_existent.get_available_tables() == []

    # Create dummy tables
    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "edstays.csv", index=False)
    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "diagnosis.csv", index=False)
    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "medrecon.csv", index=False)
    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "pyxis.csv", index=False)
    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "triage.csv", index=False)
    pd.DataFrame({"stay_id": [1]}).to_csv(tmp_path / "vitalsign.csv", index=False)

    available = loader.get_available_tables()
    assert len(available) == 6

    # Test all dedicated loader methods
    assert len(loader.load_edstays()) == 1
    assert len(loader.load_diagnosis()) == 1
    assert len(loader.load_medrecon()) == 1
    assert len(loader.load_pyxis()) == 1
    assert len(loader.load_triage()) == 1
    assert len(loader.load_vitalsign()) == 1

    all_tables = loader.load_all_available()
    assert len(all_tables) == 6


# ==============================================================================
# Validator Unit Tests
# ==============================================================================


def test_check_non_empty() -> None:
    """Test check_non_empty accepts populated df and rejects empty df."""
    valid_df = pd.DataFrame({"col": [1, 2]})
    check_non_empty(valid_df, "TestTable")  # Should not raise

    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError) as exc:
        check_non_empty(empty_df, "TestTable")
    assert "empty" in str(exc.value)


def test_check_required_columns() -> None:
    """Test check_required_columns passes when present and fails when missing."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    check_required_columns(df, ["a", "b"])  # Should not raise

    with pytest.raises(DataValidationError) as exc:
        check_required_columns(df, ["a", "d", "e"], "TestTable")
    assert "Missing required columns" in str(exc.value)
    assert "'d'" in str(exc.value)


def test_check_duplicate_identifiers() -> None:
    """Test check_duplicate_identifiers detects repeated keys."""
    valid_df = pd.DataFrame({"stay_id": [1, 2, 3]})
    check_duplicate_identifiers(valid_df, "stay_id")  # Should not raise

    dup_df = pd.DataFrame({"stay_id": [1, 2, 2, 3]})
    with pytest.raises(DataValidationError) as exc:
        check_duplicate_identifiers(dup_df, "stay_id")
    assert "Found 1 duplicate values" in str(exc.value)

    # Missing ID column
    with pytest.raises(DataValidationError) as exc_missing:
        check_duplicate_identifiers(valid_df, "nonexistent_id")
    assert "not found" in str(exc_missing.value)


def test_check_datetime_columns() -> None:
    """Test check_datetime_columns parses valid dates and raises on bad data."""
    valid_df = pd.DataFrame({"time_a": ["2026-01-01 10:00:00", "2026-01-02 12:30:00"]})
    # Also pass a non-existent column name which should be skipped gracefully
    check_datetime_columns(valid_df, ["time_a", "missing_time_col"])

    invalid_df = pd.DataFrame({"time_a": ["2026-01-01 10:00:00", "not_a_valid_date"]})
    with pytest.raises(DataValidationError) as exc:
        check_datetime_columns(invalid_df, ["time_a"])
    assert "unparseable datetime" in str(exc.value)


def test_check_missing_values() -> None:
    """Test check_missing_values reports counts and enforces critical rules."""
    df = pd.DataFrame(
        {
            "col_complete": [1, 2, 3],
            "col_sparse": [1, None, 3],
            "col_empty": [None, None, None],
        }
    )

    missing = check_missing_values(df)
    assert missing["col_sparse"] == 1
    assert missing["col_empty"] == 3
    assert "col_complete" not in missing

    # Critical columns check
    with pytest.raises(DataValidationError) as exc:
        check_missing_values(df, critical_columns=["col_sparse"])
    assert "Critical column 'col_sparse' contains 1 missing" in str(exc.value)


def test_check_time_ordering() -> None:
    """Test check_time_ordering enforces start <= end."""
    valid_df = pd.DataFrame(
        {
            "intime": ["2026-01-01 10:00:00", "2026-01-01 12:00:00"],
            "outtime": ["2026-01-01 11:00:00", "2026-01-01 12:00:00"],
        }
    )
    assert check_time_ordering(valid_df, "intime", "outtime") == 0

    invalid_df = pd.DataFrame(
        {
            "intime": ["2026-01-01 12:00:00"],
            "outtime": ["2026-01-01 10:00:00"],
        }
    )
    with pytest.raises(DataValidationError) as exc:
        check_time_ordering(invalid_df, "intime", "outtime")
    assert "precedes" in str(exc.value)


def test_validate_ed_stays_valid() -> None:
    """Test comprehensive validate_ed_stays on valid encounter data."""
    valid_df = pd.DataFrame(
        {
            "stay_id": [101, 102],
            "subject_id": [1, 2],
            "intime": ["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
            "outtime": ["2026-01-01 12:00:00", "2026-01-01 14:00:00"],
            "acuity": [2.0, 3.0],
        }
    )
    report = validate_ed_stays(valid_df)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert len(report["warnings"]) == 0


def test_validate_ed_stays_empty() -> None:
    """Test validate_ed_stays raises DataValidationError on empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_ed_stays(empty_df)


def test_validate_ed_stays_missing_columns() -> None:
    """Test validate_ed_stays fails when required fields are missing."""
    invalid_df = pd.DataFrame({"stay_id": [101, 102]})
    with pytest.raises(DataValidationError) as exc:
        validate_ed_stays(invalid_df)
    assert "Missing required columns" in str(exc.value)


def test_validate_ed_stays_duplicates() -> None:
    """Test validate_ed_stays fails on duplicate stay_id."""
    dup_df = pd.DataFrame(
        {
            "stay_id": [101, 101],
            "subject_id": [1, 2],
            "intime": ["2026-01-01 10:00:00", "2026-01-01 11:00:00"],
            "outtime": ["2026-01-01 12:00:00", "2026-01-01 14:00:00"],
            "acuity": [2.0, 3.0],
        }
    )
    with pytest.raises(DataValidationError) as exc:
        validate_ed_stays(dup_df)
    assert "duplicate" in str(exc.value).lower()


def test_validate_ed_stays_invalid_times() -> None:
    """Test validate_ed_stays fails when outtime precedes intime."""
    inverted_df = pd.DataFrame(
        {
            "stay_id": [101],
            "subject_id": [1],
            "intime": ["2026-01-01 14:00:00"],
            "outtime": ["2026-01-01 10:00:00"],
            "acuity": [2.0],
        }
    )
    with pytest.raises(DataValidationError) as exc:
        validate_ed_stays(inverted_df)
    assert "precedes" in str(exc.value).lower()


def test_validate_ed_stays_missing_acuity_warning() -> None:
    """Test missing acuity logs a warning without fatal failure."""
    sparse_acuity_df = pd.DataFrame(
        {
            "stay_id": [101],
            "subject_id": [1],
            "intime": ["2026-01-01 10:00:00"],
            "outtime": ["2026-01-01 12:00:00"],
            "acuity": [None],
        }
    )
    report = validate_ed_stays(sparse_acuity_df)
    assert report["is_valid"] is True
    assert len(report["warnings"]) == 1
    assert "missing acuity" in report["warnings"][0]
