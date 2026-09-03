from pathlib import Path

import pandas as pd

from trading_ml.data.normalization import normalize_fi2010
from trading_ml.data.parquet import read_parquet, write_parquet
from trading_ml.data.validation import validate_market_data


def test_lob_pipeline_end_to_end(tmp_path):
    raw = pd.DataFrame(
        [
            [
                101.0, 10.0, 99.0, 12.0,
            ] + [0.0] * 36,
            [
                102.0, 11.0, 100.0, 13.0,
            ] + [0.0] * 36,
        ]
    )

    normalized = normalize_fi2010(
        raw,
        symbol="TEST",
    )

    validation = pd.DataFrame(
        {
            "timestamp": normalized["event_id"],
            "bid_price_1": normalized["bid_price_1"],
            "bid_size_1": normalized["bid_size_1"],
            "ask_price_1": normalized["ask_price_1"],
            "ask_size_1": normalized["ask_size_1"],
        }
    )

    validate_market_data(validation)

    output = tmp_path / "normalized.parquet"

    write_parquet(
        normalized,
        output,
    )

    loaded = read_parquet(output)

    assert len(loaded) == 2
    assert loaded["symbol"].unique().tolist() == ["TEST"]
    assert loaded["bid_price_1"].tolist() == [99.0, 100.0]