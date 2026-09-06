from __future__ import annotations

import math
from typing import Generator

import numpy as np
import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge_window: int = 10,
    embargo_window: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-ordered dataset into Train, Validation, and Test sets sequentially
    with purging and embargo to prevent target overlap leakage across set boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered DataFrame.
    train_ratio : float, default=0.70
        Fraction of dataset allocated to training.
    val_ratio : float, default=0.15
        Fraction of dataset allocated to validation.
    test_ratio : float, default=0.15
        Fraction of dataset allocated to testing.
    purge_window : int, default=10
        Number of observations before the set boundary to purge (e.g. target horizon H).
    embargo_window : int, default=0
        Number of observations after the set boundary to embargo.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df) subsets with resetting or preserving index.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, abs_tol=1e-5):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("All split ratios must be strictly positive")

    if purge_window < 0 or embargo_window < 0:
        raise ValueError("Purge and embargo windows must be non-negative")

    n = len(df)
    if n == 0:
        raise ValueError("Cannot split empty DataFrame")

    train_end_raw = int(n * train_ratio)
    val_end_raw = int(n * (train_ratio + val_ratio))

    # Apply purging before boundary and embargo after boundary
    train_end_purged = max(0, train_end_raw - purge_window)
    val_start_embargoed = min(n, train_end_raw + embargo_window)

    val_end_purged = max(val_start_embargoed, val_end_raw - purge_window)
    test_start_embargoed = min(n, val_end_raw + embargo_window)

    train_df = df.iloc[:train_end_purged].copy()
    val_df = df.iloc[val_start_embargoed:val_end_purged].copy()
    test_df = df.iloc[test_start_embargoed:].copy()

    return train_df, val_df, test_df


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation generator for time-series market data.

    Generates expanding window train/validation index splits with purging
    to eliminate target leakage across fold boundaries.

    Parameters
    ----------
    n_splits : int, default=4
        Number of walk-forward folds.
    purge_window : int, default=10
        Number of observations prior to validation fold to purge.
    embargo_window : int, default=0
        Number of observations at the start of validation fold to embargo.
    min_train_ratio : float, default=0.4
        Initial training set fraction for the first fold.
    """

    def __init__(
        self,
        n_splits: int = 4,
        purge_window: int = 10,
        embargo_window: int = 0,
        min_train_ratio: float = 0.4,
    ) -> None:
        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}")
        if purge_window < 0 or embargo_window < 0:
            raise ValueError("Purge and embargo windows must be non-negative")
        if not (0.0 < min_train_ratio < 1.0):
            raise ValueError("min_train_ratio must be between 0 and 1")

        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.min_train_ratio = min_train_ratio

    def split(
        self, df: pd.DataFrame
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train and validation index arrays.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            (train_indices, val_indices)
        """
        n_samples = len(df)
        if n_samples == 0:
            raise ValueError("Cannot split empty DataFrame")

        min_train_size = int(n_samples * self.min_train_ratio)
        val_total_samples = n_samples - min_train_size

        if val_total_samples < self.n_splits:
            raise ValueError(
                f"Dataset too small ({n_samples} rows) for {self.n_splits} splits "
                f"with min_train_ratio={self.min_train_ratio}"
            )

        fold_size = val_total_samples // self.n_splits

        for i in range(self.n_splits):
            train_end_raw = min_train_size + i * fold_size
            val_end_raw = (
                train_end_raw + fold_size if i < self.n_splits - 1 else n_samples
            )

            train_end_purged = max(0, train_end_raw - self.purge_window)
            val_start_embargoed = min(n_samples, train_end_raw + self.embargo_window)

            train_indices = np.arange(0, train_end_purged)
            val_indices = np.arange(val_start_embargoed, val_end_raw)

            if len(train_indices) == 0 or len(val_indices) == 0:
                continue

            yield train_indices, val_indices
