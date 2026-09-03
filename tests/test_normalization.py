import pandas as pd
import pytest

from trading_ml.data.normalization import normalize_fi2010


def test_normalize_fi2010_maps_first_level_correctly():
    raw = pd.DataFrame(
        [[
            101.0, 10.0, 100.0, 12.0,
        ] + [0.0] * 36]
    )

    result = normalize_fi2010(raw, symbol="TEST")

    assert result.loc[0, "ask_price_1"] == 101.0
    assert result.loc[0, "ask_size_1"] == 10.0
    assert result.loc[0, "bid_price_1"] == 100.0
    assert result.loc[0, "bid_size_1"] == 12.0


def test_normalize_fi2010_maps_tenth_level():
    values = [0.0] * 40

    values[36] = 110.0
    values[37] = 20.0
    values[38] = 109.0
    values[39] = 25.0

    raw = pd.DataFrame([values])

    result = normalize_fi2010(raw, symbol="TEST")

    assert result.loc[0, "ask_price_10"] == 110.0
    assert result.loc[0, "ask_size_10"] == 20.0
    assert result.loc[0, "bid_price_10"] == 109.0
    assert result.loc[0, "bid_size_10"] == 25.0


def test_normalize_fi2010_rejects_wrong_column_count():
    raw = pd.DataFrame([[1.0] * 39])

    with pytest.raises(ValueError, match="Expected 40"):
        normalize_fi2010(raw, symbol="TEST")


def test_normalize_fi2010_rejects_empty_dataframe():
    raw = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        normalize_fi2010(raw, symbol="TEST")


def test_event_ids_are_deterministic():
    raw = pd.DataFrame(
        [
            [1.0] * 40,
            [2.0] * 40,
            [3.0] * 40,
        ]
    )

    result = normalize_fi2010(raw, symbol="TEST")

    assert result["event_id"].tolist() == [0, 1, 2]