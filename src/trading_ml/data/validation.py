from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "bid_price_1",
    "bid_size_1",
    "ask_price_1",
    "ask_size_1",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required market-data columns are present."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_prices(df: pd.DataFrame) -> None:
    """Validate basic bid/ask price constraints."""

    if (df["bid_price_1"] <= 0).any():
        raise ValueError("Found non-positive bid prices.")

    if (df["ask_price_1"] <= 0).any():
        raise ValueError("Found non-positive ask prices.")

    if (df["bid_price_1"] > df["ask_price_1"]).any():
        raise ValueError("Found crossed order book: bid > ask.")


def validate_sizes(df: pd.DataFrame) -> None:
    """Validate that order quantities are non-negative."""

    if (df["bid_size_1"] < 0).any():
        raise ValueError("Found negative bid sizes.")

    if (df["ask_size_1"] < 0).any():
        raise ValueError("Found negative ask sizes.")


def validate_timestamps(df: pd.DataFrame) -> None:
    """Validate timestamp ordering."""

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted.")


def validate_market_data(df: pd.DataFrame) -> None:
    """Run all market-data validation checks."""

    validate_schema(df)
    validate_prices(df)
    validate_sizes(df)
    validate_timestamps(df)