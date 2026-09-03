from __future__ import annotations

import pandas as pd


def normalize_fi2010(
    raw: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    """
    Convert a FI-2010 40-feature LOB matrix into the project's
    canonical order-book representation.

    FI-2010 stores each level as:

        ask_price, ask_volume, bid_price, bid_volume

    for 10 levels.

    Parameters
    ----------
    raw:
        DataFrame containing the 40 raw LOB columns.
        Columns must be ordered according to the FI-2010 convention.

    symbol:
        Instrument identifier assigned to the dataset.

    Returns
    -------
    pd.DataFrame
        Canonical long-column market-data representation.
    """
    normalized = pd.DataFrame(index=raw.index)

    normalized["event_id"] = range(len(raw))
    normalized["symbol"] = symbol
    if raw.empty:
        raise ValueError("Input FI-2010 dataframe is empty.")

    if raw.shape[1] != 40:
        raise ValueError(
            f"Expected 40 raw LOB columns, got {raw.shape[1]}."
        )

    normalized = pd.DataFrame(index=raw.index)
    normalized['event_id'] = range(len(raw))
    normalized["symbol"] = symbol

    for level in range(1, 11):
        base = (level - 1) * 4

        normalized[f"ask_price_{level}"] = raw.iloc[:, base]
        normalized[f"ask_size_{level}"] = raw.iloc[:, base + 1]

        normalized[f"bid_price_{level}"] = raw.iloc[:, base + 2]
        normalized[f"bid_size_{level}"] = raw.iloc[:, base + 3]

    return normalized.reset_index(drop=True)