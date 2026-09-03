import pandas as pd
import pytest

from trading_ml.features.microstructure import (
    add_depth_features,
    add_microprice,
    add_mid_price,
    add_order_book_imbalance,
    add_spread,
    build_microstructure_features,
)


def make_lob_dataframe() -> pd.DataFrame:
    data = {
        "bid_price_1": [100.0],
        "ask_price_1": [101.0],
        "bid_size_1": [90.0],
        "ask_size_1": [10.0],
    }

    for level in range(2, 11):
        data[f"bid_price_{level}"] = [99.0 - level]
        data[f"ask_price_{level}"] = [102.0 + level]
        data[f"bid_size_{level}"] = [10.0]
        data[f"ask_size_{level}"] = [10.0]

    return pd.DataFrame(data)


def test_mid_price():
    df = make_lob_dataframe()

    result = add_mid_price(df)

    assert result.loc[0, "mid_price"] == pytest.approx(100.5)


def test_spread():
    df = make_lob_dataframe()

    result = add_mid_price(df)
    result = add_spread(result)

    assert result.loc[0, "spread"] == pytest.approx(1.0)
    assert result.loc[0, "relative_spread"] == pytest.approx(
        1.0 / 100.5
    )


def test_level_one_imbalance():
    df = make_lob_dataframe()

    result = add_order_book_imbalance(
        df,
        levels=(1,),
    )

    expected = (90.0 - 10.0) / (90.0 + 10.0)

    assert result.loc[0, "obi_1"] == pytest.approx(
        expected
    )


def test_level_three_imbalance():
    df = make_lob_dataframe()

    result = add_order_book_imbalance(
        df,
        levels=(3,),
    )

    bid_depth = 90.0 + 10.0 + 10.0
    ask_depth = 10.0 + 10.0 + 10.0

    expected = (
        (bid_depth - ask_depth)
        / (bid_depth + ask_depth)
    )

    assert result.loc[0, "obi_3"] == pytest.approx(
        expected
    )


def test_microprice():
    df = make_lob_dataframe()

    result = add_mid_price(df)
    result = add_microprice(result)

    expected = (
        (101.0 * 90.0 + 100.0 * 10.0)
        / 100.0
    )

    assert result.loc[0, "microprice"] == pytest.approx(
        expected
    )


def test_depth_features():
    df = make_lob_dataframe()

    result = add_depth_features(df)

    assert result.loc[0, "bid_depth_5"] == pytest.approx(
        90.0 + 4 * 10.0
    )

    assert result.loc[0, "ask_depth_5"] == pytest.approx(
        10.0 + 4 * 10.0
    )

    assert result.loc[0, "depth_ratio_5"] == pytest.approx(
        130.0 / 50.0
    )


def test_complete_feature_pipeline():
    df = make_lob_dataframe()

    result = build_microstructure_features(df)

    expected_columns = {
        "mid_price",
        "spread",
        "relative_spread",
        "obi_1",
        "obi_3",
        "obi_5",
        "obi_10",
        "microprice",
        "bid_depth_5",
        "ask_depth_5",
        "depth_ratio_5",
    }

    assert expected_columns.issubset(result.columns)