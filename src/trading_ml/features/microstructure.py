from __future__ import annotations

import numpy as np
import pandas as pd


def add_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """Add the best-bid/best-ask midpoint."""

    result = df.copy()

    result["mid_price"] = (
        result["bid_price_1"] +
        result["ask_price_1"]
    ) / 2.0

    return result


def add_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and relative bid-ask spread."""

    result = df.copy()

    result["spread"] = (
        result["ask_price_1"] -
        result["bid_price_1"]
    )

    result["relative_spread"] = (
        result["spread"] /
        result["mid_price"]
    )

    return result


def add_order_book_imbalance(
    df: pd.DataFrame,
    levels: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """
    Add order-book imbalance for multiple depth levels.

    OBI = (bid_volume - ask_volume)
          / (bid_volume + ask_volume)
    """

    result = df.copy()

    for level in levels:
        bid_columns = [
            f"bid_size_{i}"
            for i in range(1, level + 1)
        ]

        ask_columns = [
            f"ask_size_{i}"
            for i in range(1, level + 1)
        ]

        bid_depth = result[bid_columns].sum(axis=1)
        ask_depth = result[ask_columns].sum(axis=1)

        denominator = bid_depth + ask_depth

        result[f"obi_{level}"] = np.where(
            denominator > 0,
            (bid_depth - ask_depth) / denominator,
            0.0,
        )

    return result


def add_microprice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the level-1 microprice.

    Microprice weights the best bid and ask according
    to the opposing queue sizes.
    """

    result = df.copy()

    bid_price = result["bid_price_1"]
    ask_price = result["ask_price_1"]

    bid_size = result["bid_size_1"]
    ask_size = result["ask_size_1"]

    denominator = bid_size + ask_size

    result["microprice"] = np.where(
        denominator > 0,
        (
            ask_price * bid_size +
            bid_price * ask_size
        ) / denominator,
        result["mid_price"],
    )

    return result


def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregate liquidity/depth features."""

    result = df.copy()

    bid_columns = [
        f"bid_size_{i}"
        for i in range(1, 6)
    ]

    ask_columns = [
        f"ask_size_{i}"
        for i in range(1, 6)
    ]

    result["bid_depth_5"] = result[bid_columns].sum(axis=1)
    result["ask_depth_5"] = result[ask_columns].sum(axis=1)

    result["depth_ratio_5"] = np.where(
        result["ask_depth_5"] > 0,
        result["bid_depth_5"] /
        result["ask_depth_5"],
        np.nan,
    )

    return result


def build_microstructure_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete microstructure feature set.

    All features use information available at time t.
    No future observations are used.
    """

    result = add_mid_price(df)
    result = add_spread(result)
    result = add_order_book_imbalance(result)
    result = add_microprice(result)
    result = add_depth_features(result)

    return result