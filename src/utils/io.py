"""Utility helpers for loading and saving experiment data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .logger import get_logger

LOGGER = get_logger("utils.io")


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory of *path* if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a :class:`~pandas.DataFrame` to ``path`` as CSV."""

    ensure_parent_dir(path)
    df.to_csv(path, index=index)
    LOGGER.info("Wrote %s rows to %s", len(df), path)


def read_dataframe(path: Path) -> pd.DataFrame:
    """Load a CSV file into a dataframe."""

    df = pd.read_csv(path)
    LOGGER.info("Loaded %s rows from %s", len(df), path)
    return df


def write_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Serialise ``data`` as JSON."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)
    LOGGER.info("Wrote JSON file to %s", path)


def load_json(path: Path) -> Any:
    """Read JSON data from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    LOGGER.info("Loaded JSON file from %s", path)
    return data


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Save a DataFrame to CSV file (alias for write_dataframe)."""
    write_dataframe(df, path, index=index)


def list_files(directory: Path, pattern: str = "*.csv") -> list[Path]:
    """Return a sorted list of files under ``directory`` matching ``pattern``."""

    return sorted(directory.glob(pattern))


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file (alias for read_dataframe)."""
    return read_dataframe(path)
