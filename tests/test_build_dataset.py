from pathlib import Path

import pandas as pd
import pytest

from trading_ml.data.parquet import read_parquet
from trading_ml.data.pipeline import build_processed_dataset


def make_raw_lob_dataframe(rows: int = 25) -> pd.DataFrame:
    records = []
    for i in range(rows):
        row = [
            100.0 + i * 0.1,  # ask_price_1
            10.0,             # ask_size_1
            99.9 + i * 0.1,   # bid_price_1
            10.0,             # bid_size_1
        ]
        # Rest of 40 columns
        for level in range(2, 11):
            row.extend([102.0 + level, 5.0, 98.0 - level, 5.0])
        records.append(row)
    return pd.DataFrame(records)


def test_build_processed_dataset_pipeline():
    raw_df = make_raw_lob_dataframe(25)

    # With horizon=5 and drop_na_targets=True, 25 - 5 = 20 rows
    processed = build_processed_dataset(
        raw_df,
        symbol="BTCUSDT",
        horizon=5,
        threshold=0.0001,
        drop_na_targets=True,
    )

    assert len(processed) == 20
    assert "mid_price" in processed.columns
    assert "obi_1" in processed.columns
    assert "future_return" in processed.columns
    assert "target" in processed.columns
    assert processed["target"].notna().all()


def test_build_processed_dataset_keep_na_targets():
    raw_df = make_raw_lob_dataframe(25)

    processed = build_processed_dataset(
        raw_df,
        symbol="BTCUSDT",
        horizon=5,
        threshold=0.0001,
        drop_na_targets=False,
    )

    assert len(processed) == 25
    assert processed["target"].iloc[-5:].isna().all()
