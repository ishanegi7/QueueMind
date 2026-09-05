"""
Data loading module for MIMIC-IV-ED emergency department datasets.

Provides modular and configurable utilities to locate, validate, and load
authorized MIMIC-IV-ED tables from CSV, Gzipped CSV, or Apache Parquet formats.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Standard MIMIC-IV-ED tables supported by QueueMind
SUPPORTED_TABLES: Sequence[str] = (
    "edstays",
    "diagnosis",
    "medrecon",
    "pyxis",
    "triage",
    "vitalsign",
)

# Supported file extensions in priority order
SUPPORTED_EXTENSIONS: Sequence[str] = (
    ".parquet",
    ".csv.gz",
    ".csv",
)


def get_data_dir(
    env_var: str = "MIMIC_DATA_DIR",
    default: Union[str, Path] = "data/raw",
) -> Path:
    """
    Retrieve the configured MIMIC-IV-ED raw data directory.

    Checks the specified environment variable first, falling back to the
    default path if not set. Uses pathlib to ensure cross-platform compatibility.

    Args:
        env_var: The environment variable name to read.
        default: Fallback directory path if env_var is unset.

    Returns:
        Path: Path object pointing to the data directory.
    """
    data_dir_str = os.environ.get(env_var)
    if data_dir_str:
        return Path(data_dir_str)
    return Path(default)


def find_table_file(table_name: str, data_dir: Path) -> Optional[Path]:
    """
    Locate an expected MIMIC-IV-ED table file in the given data directory.

    Checks supported extensions in order (.parquet, .csv.gz, .csv).

    Args:
        table_name: Name of the table (e.g. 'edstays', 'triage').
        data_dir: Path to the directory containing raw data files.

    Returns:
        Optional[Path]: The path to the first matching file, or None if not found.
    """
    for ext in SUPPORTED_EXTENSIONS:
        candidate = data_dir / f"{table_name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def load_mimic_table(
    table_name: str,
    data_dir: Optional[Union[str, Path]] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Load a single MIMIC-IV-ED table into a pandas DataFrame.

    Supports .parquet, .csv.gz, and .csv formats. Uses efficient parsers and
    provides descriptive error diagnostics if the file is missing or invalid.

    Args:
        table_name: Name of the table to load (e.g., 'edstays', 'vitalsign').
        data_dir: Optional directory path. Defaults to get_data_dir() if None.
        **kwargs: Additional keyword arguments forwarded to the pandas reader.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If no supported file for table_name exists in data_dir.
        RuntimeError: If reading the file fails.
    """
    resolved_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    file_path = find_table_file(table_name, resolved_dir)

    if file_path is None:
        msg = (
            f"Table '{table_name}' not found in directory '{resolved_dir}'. "
            f"Expected one of: {[f'{table_name}{ext}' for ext in SUPPORTED_EXTENSIONS]}"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading table '%s' from %s", table_name, file_path)

    try:
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path, **kwargs)
        # pd.read_csv handles both .csv and .csv.gz automatically
        return pd.read_csv(file_path, **kwargs)
    except Exception as exc:
        msg = f"Failed to read table '{table_name}' from '{file_path}': {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc


class MIMICDataLoader:
    """
    Configurable data loader manager for the MIMIC-IV-ED dataset.

    Provides dedicated methods for all core tables and inspections of
    available data files in the target directory.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the loader with a specified or default data directory.

        Args:
            data_dir: Optional directory path containing MIMIC-IV-ED files.
        """
        self.data_dir: Path = Path(data_dir) if data_dir is not None else get_data_dir()

    def get_available_tables(self) -> List[str]:
        """
        Discover which supported MIMIC-IV-ED tables exist in the data directory.

        Returns:
            List[str]: Table names found in the data directory.
        """
        if not self.data_dir.is_dir():
            return []

        available: List[str] = []
        for table in SUPPORTED_TABLES:
            if find_table_file(table, self.data_dir) is not None:
                available.append(table)
        return available

    def load_table(self, table_name: str, **kwargs: object) -> pd.DataFrame:
        """Load any supported table by name."""
        return load_mimic_table(table_name, data_dir=self.data_dir, **kwargs)

    def load_edstays(self, **kwargs: object) -> pd.DataFrame:
        """Load the master edstays table."""
        return self.load_table("edstays", **kwargs)

    def load_diagnosis(self, **kwargs: object) -> pd.DataFrame:
        """Load the diagnosis billing codes table."""
        return self.load_table("diagnosis", **kwargs)

    def load_medrecon(self, **kwargs: object) -> pd.DataFrame:
        """Load the home medication reconciliation table."""
        return self.load_table("medrecon", **kwargs)

    def load_pyxis(self, **kwargs: object) -> pd.DataFrame:
        """Load the automated medication dispensing (pyxis) table."""
        return self.load_table("pyxis", **kwargs)

    def load_triage(self, **kwargs: object) -> pd.DataFrame:
        """Load the emergency department triage assessment table."""
        return self.load_table("triage", **kwargs)

    def load_vitalsign(self, **kwargs: object) -> pd.DataFrame:
        """Load the longitudinal vital signs table."""
        return self.load_table("vitalsign", **kwargs)

    def load_all_available(self) -> Dict[str, pd.DataFrame]:
        """
        Load all supported tables that currently exist in the data directory.

        Returns:
            Dict[str, pd.DataFrame]: Mapping from table name to loaded DataFrame.
        """
        tables = self.get_available_tables()
        logger.info("Found %d available tables to load: %s", len(tables), tables)
        return {tbl: self.load_table(tbl) for tbl in tables}
