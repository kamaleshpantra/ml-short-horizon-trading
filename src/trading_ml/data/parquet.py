from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
) -> Path:
    """
    Write a DataFrame to a Parquet file.

    The parent directory is created automatically.
    """

    if df.empty:
        raise ValueError("Cannot write an empty DataFrame.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        destination,
        index=False,
    )

    return destination


def read_parquet(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read a Parquet dataset from disk.
    """

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {source}"
        )

    return pd.read_parquet(source)