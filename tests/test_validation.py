import pandas as pd
import pytest

from trading_ml.data.validation import validate_market_data


def test_valid_market_data():

    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "bid_price_1": [100.0, 101.0, 102.0],
            "bid_size_1": [10.0, 12.0, 15.0],
            "ask_price_1": [100.1, 101.1, 102.1],
            "ask_size_1": [8.0, 11.0, 13.0],
        }
    )

    validate_market_data(df)


def test_crossed_order_book():

    df = pd.DataFrame(
        {
            "timestamp": [1],
            "bid_price_1": [101.0],
            "bid_size_1": [10.0],
            "ask_price_1": [100.0],
            "ask_size_1": [8.0],
        }
    )

    with pytest.raises(ValueError):
        validate_market_data(df)


def test_negative_quantity():

    df = pd.DataFrame(
        {
            "timestamp": [1],
            "bid_price_1": [100.0],
            "bid_size_1": [-10.0],
            "ask_price_1": [100.1],
            "ask_size_1": [8.0],
        }
    )

    with pytest.raises(ValueError):
        validate_market_data(df)